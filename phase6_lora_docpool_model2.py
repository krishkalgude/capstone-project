import os
import re
import sys
import json
import math
import time
import pickle
import random
import zipfile
import shutil
import hashlib
import functools
import contextlib
import subprocess
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# =====================================================================================
# CONFIG
# =====================================================================================
# ------------------------------------------------------------------
# SWITCH: run once with True (tiny subset, 2 epochs, simulated interruption + resume, ~10-15 min) to verify the whole
# pipeline end to end, then set False for the real run.
# ------------------------------------------------------------------
SMOKE_TEST = False

RUN_TAG = "p6_lora_docpool_m2_v1" + ("_smoke" if SMOKE_TEST else "")
IS_KAGGLE = os.path.isdir("/kaggle/working")
INPUT_ROOT = "/kaggle/input"
WORK_DIR = ("/kaggle/working" if IS_KAGGLE else ".") + f"/{RUN_TAG}"
CKPT_DIR = f"{WORK_DIR}/checkpoints"
CACHE_DIR = f"{WORK_DIR}/cache"
CACHE_PATH = f"{CACHE_DIR}/prepared_chunks_docpool_v1.pkl"
OUT_DIR = f"{WORK_DIR}/submission"

# ---- Phase-1 artefacts (None = auto-search /kaggle/input) ----
PHASE1_CKPT = None            # path to best.pt of Model 2
PHASE1_TOKENIZER_DIR = None   # folder with tokenizer.json (+ meta.json) from the BPE notebook
PHASE1_TAG_HINT = "decoder_only_rope_gqa_swiglu"

# ---- data ----
HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
PHASE6_FILES = {"train": "Phase_6_Summ/Train/train.json", "test": "Phase_6_Summ/Test/test.json"}
VAL_DOCS = 150                 # held-out documents (no public dev split exists)
MIN_SUMMARY_WORDS = 20         # drop empty / degenerate reference summaries
MAX_TRAIN_SUMMARY_DOC_RATIO = 0.9   # drop pairs whose summary is almost as long as the document

# ---- chunking / alignment ----
CONTEXT_LENGTH = 1024          # the context the Phase-1 model was trained with
CHUNK_TOKENS = 560             # source tokens per chunk
TGT_MAX_TOKENS = 420           # max summary tokens per chunk (incl. the end marker)
MAX_SENT_WORDS = 200           # run-on "sentences" are cut into pieces of at most this many words
MIN_ALIGN_SCORE = 0.10         # below this, a summary sentence is placed by relative position instead
ALIGN_TIE_EPS = 0.02           # chunks within this score of the best are ties -> pick closest to previous
KEEP_EMPTY_P = 0.30            # share of "nothing to summarise" chunks kept as (chunk -> empty) samples
USE_PART_TAG = True            # prefix "[Part k/n]" so the model knows where in the document it is
SEP_TEXT = "\n\nSUMMARY:\n"
END_TEXT = "\n\n"              # end-of-summary marker (the Phase-1 EOS embedding was never trained)
MAX_TRAIN_CHUNKS = None        # e.g. 30000 to cap the training set for a quick run

# ---- learned document-level pooling (the long-context method of this file) ----
N_DOC_TOK = 16                 # soft document tokens put in front of every chunk prompt (they use N_DOC_TOK of the 1024 context tokens)
POOL_WIN = 32                  # frozen-model token states are average-pooled in windows of this many tokens -> one "slot" per window
                               #   cache ~ (document tokens / POOL_WIN) x 640 x 2 bytes = ~1.7 GB for the 7k training documents at 32
                               #   (64: half the size and half the resolution of the pooling, 16: twice both)
POOL_DIM = 512                 # width of the pooling attention
POOL_HEADS = 8
POOL_FF = 1024
POOL_DROPOUT = 0.1
POOL_SLOT_DROP = 0.10          # training only: every slot is hidden from the pooling with this probability (regulariser)
POOL_EXCLUDE_OWN = True        # the slots of the chunk being summarised are not pooled (the decoder reads that chunk itself)
POOL_MAX_DIST = 8              # embedding of the signed chunk distance to the current chunk (clamped to +-POOL_MAX_DIST)
CTX_LR_MULT = 2.0              # the (new, randomly initialised) context module learns this much faster than the LoRA matrices
CTX_GATE_INIT = 0.5            # scale of the learned part of the document tokens at the start (relative to a token embedding)
ENC_TOKEN_BUDGET = 12000       # padded tokens per forward pass of the frozen model while the document features are extracted
MIN_CHUNK_TOKENS = 256         # init_state refuses a setup in which the document tokens leave less chunk text than this

# ---- LoRA ----
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "w_gate", "w_up", "w_down")
TRAIN_NORMS = False            # also train the RMSNorm gains (tiny)
FT_DROPOUT = 0.1               # residual dropout of the frozen backbone while fine-tuning (Phase 1 used 0.2)

# ---- optimisation ----
EPOCHS = 10                    # TOTAL number of epochs (a resume continues from the checkpoint's epoch up to this number)
MICRO_BS = 4                   # lower to 2 (raise ACCUM) on CUDA OOM
ACCUM = 8
LR = 3e-4
MIN_LR_RATIO = 0.05
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
USE_EMA = True                 # exponential moving average of the LoRA weights; validation + best adapter use the EMA weights
EMA_DECAY = 0.998
EVAL_BS = 8
LOG_EVERY = 25
SAVE_EVERY = 100               # optimizer steps between mid-epoch checkpoints (lora_latest.pt is only ~50 MB)
SEED = 1234

# ---- time budget (a Kaggle GPU session is limited to 9 h) ----
TIME_BUDGET_H = 9.0            # wall-clock limit of THIS session (data prep + training + prediction)
FINAL_RESERVE_H = 1.0          # kept free for validation generation + test prediction at the end of the session
TIME_AWARE_LR = False          # False: cosine over exactly EPOCHS (the session just stops at the time budget; the next session
                               #        resumes the SAME cosine -> exactly EPOCHS epochs over as many sessions as needed)
                               # True : the LR also decays with wall-clock time so the cosine ends inside TRAIN_HOURS,
                               #        i.e. one session may finish with fewer than EPOCHS epochs but a fully annealed LR
TRAIN_HOURS = 7.0              # (TIME_AWARE_LR only) TOTAL training time of a FRESH run over all sessions

# ---- resuming / continuing (attach the earlier output as an input; keep RUN_TAG and all other inputs) ----
REQUIRE_RESUME = False         # True: raise a clear error if no earlier checkpoint is found instead of training from scratch
CONT_NEW_LR_CYCLE = True       # a resume whose EPOCHS differs from the checkpoint's schedule starts a NEW cycle: re-warm-up -> cosine to EPOCHS
CONT_LR_PEAK = 0.5             # peak of that new cycle = this fraction of LR (the model is already trained)
CONT_WARMUP_STEPS = 100
CONT_TRAIN_HOURS = 24.0        # (TIME_AWARE_LR only) training-time cap of a continuation over all its sessions
RESUME_CHECK = True            # after loading, re-score the restored (EMA) weights: must match the validation loss logged when saved
RESUME_CHECK_TOL = 0.01        # tolerated difference of that validation loss

# ---- inference ----
GEN_BS = 48                    # chunks decoded in parallel
GEN_MAX_NEW = TGT_MAX_TOKENS
NO_REPEAT_NGRAM = 5            # in BPE tokens
REP_PENALTY = 1.0              # 1.0 = off (repeating names / sections is normal in legal summaries)
DEDUP_JACCARD = 0.80           # drop a sentence whose word set overlaps an earlier one this much
MAX_SUMMARY_WORDS = 3000       # hard cap on the stitched summary
MAX_SUMMARY_DOC_RATIO = 0.45   # ... and cap relative to document length (reference p95 ~ 0.44)
VAL_GEN_DOCS = 60              # held-out docs used for the end-to-end ROUGE / BERTScore check
SAVE_MERGED = False            # also save the merged (LoRA folded in) model weights

# ---- what to run ----
RUN_TRAIN = True
RUN_VAL_GEN = True
RUN_PREDICT = True
USE_BERTSCORE = True           # roberta-large BERTScore on the validation generations (needs internet; skipped in the smoke test)

# ---- smoke test overrides (must come after the settings they replace) ----
if SMOKE_TEST:
    N_SMOKE = {"train": 40, "val": 8, "test": 4}   # documents
    SMOKE_MAX_DOC_WORDS = 6000                     # keep the smoke documents short
    SMOKE_STOP_STEP = 13                           # session 1 is cut after this many optimizer steps (mid-epoch)
    EPOCHS, ACCUM, SAVE_EVERY, LOG_EVERY, WARMUP_STEPS = 2, 2, 5, 2, 5
    VAL_GEN_DOCS, GEN_BS, MAX_TRAIN_CHUNKS = 3, 16, None
    USE_BERTSCORE, REQUIRE_RESUME, RESUME_CHECK = False, False, True
    TIME_AWARE_LR, SAVE_MERGED = False, False
else:
    N_SMOKE = None
EFF_BS = MICRO_BS * ACCUM

AMP = {"enabled": False, "dtype": torch.float16}
STATE = {}                     # tokenizer-derived constants, filled by init_state()
WORD_RE = re.compile(r"[a-z0-9]+")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
END_PUNCT = re.compile(r"[.!?][\"')\]]*$")


def amp_ctx():
    return torch.autocast(device_type="cuda", dtype=AMP["dtype"], enabled=AMP["enabled"])


def log(*a):
    print(*a, flush=True)


def fmt_time(sec):
    sec = int(max(sec, 0))
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):                       # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def walk_find(filename, roots):
    hits = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for d, _, files in os.walk(root):
            if filename in files:
                hits.append(d)
    return hits


# =====================================================================================
# DATA LOADING (same Hugging Face source as the Phase-6 notebook)
# =====================================================================================
def hf_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HF")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF")
    except Exception:
        return None


def load_records(split):
    from huggingface_hub import hf_hub_download
    filename = PHASE6_FILES[split]
    try:
        path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=filename, repo_type="dataset", token=hf_token())
    except Exception as e:
        cands = [filename, os.path.join(INPUT_ROOT, filename), os.path.join("/kaggle/working", filename)]
        path = next((p for p in cands if os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError(f"could not get {filename} from Hugging Face or locally: {e}") from e
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        for key in ("data", "records", "examples"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
    assert isinstance(obj, list), f"unexpected JSON structure for {split}"
    log(f"{split}: {len(obj):,} records | {path}")
    return obj


def clean_train_records(raw):
    """normalise whitespace, drop empty/degenerate summaries and duplicate documents"""
    seen, out, dropped = set(), [], Counter()
    for r in raw:
        doc = " ".join(str(r.get("document", "")).split())
        summ = " ".join(str(r.get("summary", "")).split())
        nw_s, nw_d = len(summ.split()), len(doc.split())
        if nw_s < MIN_SUMMARY_WORDS:
            dropped["short_summary"] += 1
            continue
        if nw_d == 0 or nw_s > MAX_TRAIN_SUMMARY_DOC_RATIO * nw_d:
            dropped["summary_too_long_for_doc"] += 1
            continue
        h = hashlib.md5(doc.encode("utf-8", "ignore")).hexdigest()
        if h in seen:
            dropped["duplicate_document"] += 1
            continue
        seen.add(h)
        out.append({"id": r["id"], "document": doc, "summary": summ})
    log(f"train records kept {len(out):,} / {len(raw):,} | dropped: {dict(dropped)}")
    return out


# =====================================================================================
# PHASE-1 TOKENIZER (true byte-level BPE from bpe-final.ipynb)
# =====================================================================================
class LegalByteBPE:
    def __init__(self, tokenizer_json):
        self.tokenizer = Tokenizer.from_file(tokenizer_json)
        self.vocab_size = self.tokenizer.get_vocab_size()
        vocab = self.tokenizer.get_vocab()
        self.pad_id, self.unk_id, self.eos_id = vocab["<|pad|>"], vocab["<|unk|>"], vocab["<|endoftext|>"]

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def encode_batch(self, texts):
        return [e.ids for e in self.tokenizer.encode_batch(list(texts))]

    def decode(self, ids):
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=True)


def find_tokenizer(vocab_size):
    roots = [PHASE1_TOKENIZER_DIR] if PHASE1_TOKENIZER_DIR else [INPUT_ROOT, os.getcwd(), "/kaggle/working"]
    for d in walk_find("tokenizer.json", roots):
        try:
            tok = LegalByteBPE(os.path.join(d, "tokenizer.json"))
        except Exception:
            continue
        if tok.vocab_size == vocab_size:
            log(f"tokenizer: {d} (vocab {tok.vocab_size}, pad {tok.pad_id}, unk {tok.unk_id}, eos {tok.eos_id})")
            return tok
    raise FileNotFoundError("no tokenizer.json with the right vocabulary size found - attach the BPE dataset "
                            "or set PHASE1_TOKENIZER_DIR")


def init_state(tok):
    """separator / end marker / stop-token set (every vocabulary item containing a byte-level newline)"""
    sep, end = tok.encode(SEP_TEXT), tok.encode(END_TEXT)
    vocab = tok.tokenizer.get_vocab()
    stop = sorted({i for t, i in vocab.items() if "\u010a" in t} | {tok.eos_id})   # U+010A = byte-level "\n"
    assert end and all(i in stop for i in end), "end marker must consist of newline tokens"
    mask = np.zeros(tok.vocab_size, dtype=bool)
    mask[stop] = True
    tag_max = max(len(tok.encode(f"[Part {k}/{n}]\n")) for k, n in ((999, 999), (100, 100), (9, 9)))
    room = CONTEXT_LENGTH - TGT_MAX_TOKENS - N_DOC_TOK - tag_max - len(sep)          # the document tokens share the context
    chunk_eff = min(CHUNK_TOKENS, room)
    assert chunk_eff >= MIN_CHUNK_TOKENS, "N_DOC_TOK + TGT_MAX_TOKENS leave too little room for the chunk text"
    STATE.update(sep_ids=sep, end_ids=end, stop_list=stop, stop_mask_np=mask, chunk_tokens=chunk_eff, docs=[], enc=None)
    log(f"separator {len(sep)} tokens | end marker {end} | {len(stop)} stop tokens | {N_DOC_TOK} document tokens | chunk text <= "
        f"{chunk_eff} tokens{' (reduced from ' + str(CHUNK_TOKENS) + ' to fit the document tokens)' if chunk_eff < CHUNK_TOKENS else ''} | "
        f"prompt <= {N_DOC_TOK + chunk_eff + tag_max + len(sep)}, target <= {TGT_MAX_TOKENS}, context {CONTEXT_LENGTH}")


# =====================================================================================
# PHASE-1 MODEL (verbatim architecture of Model2_clean_BPE_final.ipynb + a `features` method)
# =====================================================================================
@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 640
    n_layers: int = 8
    n_heads: int = 10
    n_kv_heads: int = 2
    ffn_hidden: int = 2048
    dropout: float = 0.2
    max_len: int = 1088
    pad_id: int = 0
    unk_id: int = 1
    eos_id: int = 2


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight).type_as(x)


def build_rope(head_dim, max_len, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len).float(), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.type_as(x)


def repeat_kv(x, groups):
    if groups == 1:
        return x
    B, KV, T, hd = x.shape
    return x[:, :, None].expand(B, KV, groups, T, hd).reshape(B, KV * groups, T, hd)


class GQAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.kv = cfg.n_heads, cfg.n_kv_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.groups = self.h // self.kv
        self.q_proj = nn.Linear(cfg.d_model, self.h * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.h * self.hd, cfg.d_model, bias=False)

    def _qkv(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        return q, k, v

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self._qkv(x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = repeat_kv(k, self.groups), repeat_kv(v, self.groups)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))

    def forward_cached(self, x, cos, sin, k_cache, v_cache, start, attn_mask):
        B, T, _ = x.shape
        q, k, v = self._qkv(x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k_cache[:, :, start:start + T] = k.to(k_cache.dtype)
        v_cache[:, :, start:start + T] = v.to(v_cache.dtype)
        keys = repeat_kv(k_cache[:, :, :start + T], self.groups).to(q.dtype)
        vals = repeat_kv(v_cache[:, :, :start + T], self.groups).to(q.dtype)
        out = F.scaled_dot_product_attention(q, keys, vals, attn_mask=attn_mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

    def forward_cached(self, x, cos, sin, cache, start, attn_mask):
        x = x + self.attn.forward_cached(self.norm1(x), cos, sin, cache["k"], cache["v"], start, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)       # also the tied output projection
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        cos, sin = build_rope(cfg.d_model // cfg.n_heads, cfg.max_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.doc_ctx = None                          # (Phase 6) learned document-context module, see attach_doc_ctx()

    def _rope(self, T, pos_ids=None):
        if pos_ids is None:
            return self.rope_cos[:T][None, None], self.rope_sin[:T][None, None]
        return self.rope_cos[pos_ids][:, None], self.rope_sin[pos_ids][:, None]

    def _embed(self, idx, prefix=None, pre_off=None):
        """token embeddings; the prefix.size(1) positions starting at pre_off (default 0) of every row are REPLACED by the
        document-context vectors prefix (B, m, D) (the prompt holds m placeholder tokens there)"""
        x = self.embed(idx)
        if prefix is not None:
            B, m = prefix.size(0), prefix.size(1)
            off = torch.zeros(B, 1, dtype=torch.long, device=idx.device) if pre_off is None else pre_off[:, None]
            rows = torch.arange(B, device=idx.device)[:, None].expand(B, m)
            cols = off + torch.arange(m, device=idx.device)[None, :]
            x = x.index_put((rows, cols), prefix.to(x.dtype))
        return x

    def features(self, idx, prefix=None, pre_off=None):
        """final hidden states (B, T, D) - lets the loss be computed on summary positions only"""
        cos, sin = self._rope(idx.size(1))
        x = self.drop(self._embed(idx, prefix, pre_off))
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.final_norm(x)

    def forward(self, idx, prefix=None, pre_off=None):
        return F.linear(self.features(idx, prefix, pre_off), self.embed.weight)

    def init_cache(self, B, total_len, dtype, device):
        kv, hd = self.cfg.n_kv_heads, self.cfg.d_model // self.cfg.n_heads
        return [{"k": torch.zeros(B, kv, total_len, hd, dtype=dtype, device=device),
                 "v": torch.zeros(B, kv, total_len, hd, dtype=dtype, device=device)} for _ in self.blocks]

    def forward_cached(self, idx, pos_ids, cache, start, attn_mask, prefix=None, pre_off=None):
        cos, sin = self._rope(idx.size(1), pos_ids)
        x = self._embed(idx, prefix, pre_off)
        for blk, c in zip(self.blocks, cache):
            x = blk.forward_cached(x, cos, sin, c, start, attn_mask)
        x = self.final_norm(x)
        return F.linear(x[:, -1], self.embed.weight)


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def find_phase1_checkpoint():
    if PHASE1_CKPT:
        return PHASE1_CKPT
    best = None
    for d in walk_find("best.pt", [INPUT_ROOT, os.getcwd(), "/kaggle/working"]):
        meta = read_json(os.path.join(d, "best_meta.json")) or {}
        tag_ok = PHASE1_TAG_HINT in str(meta.get("tag", ""))
        key = (0 if tag_ok else 1, meta.get("val_loss", 1e9))
        if best is None or key < best[0]:
            best = (key, os.path.join(d, "best.pt"))
    if best is None:
        raise FileNotFoundError("Phase-1 best.pt not found - attach the Model-2 output or set PHASE1_CKPT")
    log(f"Phase-1 checkpoint: {best[1]}")
    return best[1]


def load_phase1_model(device):
    ckpt = torch.load(find_phase1_checkpoint(), map_location="cpu", weights_only=False)
    fields = ModelConfig.__dataclass_fields__
    cfg = ModelConfig(**{k: v for k, v in ckpt["config"].items() if k in fields})
    cfg.dropout = FT_DROPOUT
    model = DecoderLM(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    log(f"Phase-1 model loaded: {count_params(model) / 1e6:.1f}M params, d_model {cfg.d_model}, "
        f"{cfg.n_layers} layers, vocab {cfg.vocab_size}, "
        f"val loss {ckpt.get('val_loss', float('nan')):.4f}")
    del ckpt
    return model.to(device), cfg


# =====================================================================================
# DOCUMENT CONTEXT (Phase 6) - the long-context method of this file
# =====================================================================================
# The FROZEN Phase-1 model reads every chunk of the document once (feature extraction, cached on disk). A small LEARNED module
# turns those features into N_DOC_TOK soft "document tokens" that sit in front of every chunk prompt:
#       [doc token 1 .. N_DOC_TOK] [part k/n] <chunk k> summary of the above: <summary of chunk k> <end>
# so the decoder (frozen Phase-1 weights + LoRA) summarises a chunk while "seeing" the rest of the document through them.
# The context module is trained together with the LoRA matrices (same loss: summary tokens only) and saved in the same adapter file.
def chunk_limit():
    return STATE["chunk_tokens"]


class Attn(nn.Module):
    """multi-head attention with an explicit key mask (own implementation, used for pooling and for the chunk-level encoders)"""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0
        self.h, self.hd = heads, dim // heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.keep = False                       # diagnostics: keep the last attention weights
        self.att = None

    def forward(self, q, k, v, allow):
        """q (B,Tq,H); k, v (B,Tk,H); allow (B,Tq or 1,Tk) bool, True = the key may be attended"""
        B, Tq, _ = q.shape
        Tk = k.size(1)
        qh = self.wq(q).view(B, Tq, self.h, self.hd).transpose(1, 2)
        kh = self.wk(k).view(B, Tk, self.h, self.hd).transpose(1, 2)
        vh = self.wv(v).view(B, Tk, self.h, self.hd).transpose(1, 2)
        sc = (qh @ kh.transpose(2, 3)).float() / math.sqrt(self.hd)
        sc = sc.masked_fill(~allow[:, None], -1e9)           # finite: a row without any allowed key stays finite (never NaN)
        att = torch.softmax(sc, dim=-1)
        if self.keep:
            self.att = att.detach()
        att = self.drop(att.to(vh.dtype))
        return self.wo((att @ vh).transpose(1, 2).reshape(B, Tq, -1))


class FFN(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class Readout(nn.Module):
    """learned queries q (B,R,H) read a set kv (B,T,H) under a key mask:  h = q + Attn(LN q, LN kv);  h + FFN(LN h)"""

    def __init__(self, dim, heads, ff, dropout):
        super().__init__()
        self.ln_q, self.ln_kv, self.ln_f = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attn(dim, heads, dropout)
        self.ffn = FFN(dim, ff, dropout)

    def forward(self, q, kv, allow):
        kvn = self.ln_kv(kv)
        h = q + self.attn(self.ln_q(q), kvn, kvn, allow)
        return h + self.ffn(self.ln_f(h))


class EncLayer(nn.Module):
    """pre-norm transformer encoder layer (self-attention under a key mask + FFN)"""

    def __init__(self, dim, heads, ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attn(dim, heads, dropout)
        self.ffn = FFN(dim, ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, allow):
        y = self.ln1(x)
        x = x + self.drop(self.attn(y, y, y, allow))
        return x + self.drop(self.ffn(self.ln2(x)))


class PrefixHead(nn.Module):
    """H-dimensional token vectors -> prefix embeddings of the decoder:   base + gate * rms(embedding) * normalise(Linear(x))
    `base` starts as the average token embedding (the decoder sees N_DOC_TOK 'average tokens' before anything is learned) and the
    learned part starts at the scale of an ordinary token embedding (times gate)."""

    def __init__(self, dim, d_model, n_tok, gate_init):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, d_model, bias=False)
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.base = nn.Parameter(torch.zeros(n_tok, d_model))
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))
        self.register_buffer("emb_rms", torch.ones(()))

    def init_from_embedding(self, emb):
        with torch.no_grad():
            e = emb.float()
            self.base.copy_(e.mean(0, keepdim=True).expand(self.base.shape[0], e.shape[1]))
            self.emb_rms.copy_(e.pow(2).mean().sqrt())

    def forward(self, x):
        y = self.out_norm(self.proj(self.norm(x)))
        return self.base[None] + self.gate * self.emb_rms * y


# ---------------------------- features of the frozen Phase-1 model (one pass over every chunk, cached) ----------------------------
def register_doc(entry):
    STATE["docs"].append(entry)
    return len(STATE["docs"]) - 1


def _enc_batches(lens):
    """chunk indices in batches of similar length, at most ENC_TOKEN_BUDGET padded tokens per forward pass"""
    order = sorted(range(len(lens)), key=lambda j: -lens[j])
    i = 0
    while i < len(order):
        nb = max(1, ENC_TOKEN_BUDGET // max(lens[order[i]], 1))
        yield order[i:i + nb]
        i += nb


@torch.no_grad()
def encode_doc(enc, chunk_ids):
    """chunk_ids: token ids of the chunks of ONE document -> the document entry (variant specific, see pool_states / finish_entry).
    The frozen Phase-1 model (eval mode, no LoRA) is causal, so right padding does not change any valid position."""
    dev = next(enc.parameters()).device
    lim, pad = chunk_limit(), enc.cfg.pad_id
    ids_all = [list(c[:lim]) or [pad] for c in chunk_ids]
    lens = [len(c) for c in ids_all]
    res = [None] * len(ids_all)
    for sel in _enc_batches(lens):
        T = max(lens[j] for j in sel)
        ids = np.full((len(sel), T), pad, dtype=np.int64)
        for r, j in enumerate(sel):
            ids[r, :lens[j]] = ids_all[j]
        ln = torch.tensor([lens[j] for j in sel], dtype=torch.long, device=dev)
        with amp_ctx():
            h = enc.features(torch.from_numpy(ids).to(dev))
        piece = pool_states(h.float(), ln)
        for r, j in enumerate(sel):
            res[j] = piece[r]
    return finish_entry(res, lens)


def load_encoder(device):
    """a second, frozen copy of the Phase-1 model: it produces the document features (the trained copy has LoRA in it)"""
    enc, _ = load_phase1_model(device)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def enc_fingerprint():
    p = find_phase1_checkpoint()
    return f"{os.path.getsize(p)}"


# ---------------------------- attaching the module to the Phase-1 model ----------------------------
def attach_doc_ctx(model, s=None):
    s = s or ctx_settings()
    dev = next(model.parameters()).device
    mod = build_ctx_module(model.cfg.d_model, s)
    mod.head.init_from_embedding(model.embed.weight.detach())
    model.doc_ctx = mod.to(dev)
    log(f"document-context module '{s['kind']}': {count_params(mod) / 1e6:.2f}M trainable parameters -> {s['n_tok']} soft tokens per chunk")


def load_model_with_ctx(device):
    model, cfg = load_phase1_model(device)
    attach_doc_ctx(model)
    return model, cfg


def batch_ctx(batch, device, docs=None):
    """context tensors of a batch of training samples (prompt, target, doc index, chunk index)"""
    return make_ctx([(b[2], b[3]) for b in batch], device, docs)


# ---------------------------- variant: LEARNED DOCUMENT-LEVEL POOLING ----------------------------
# Representation: the frozen Phase-1 model's final token states of EVERY chunk, average-pooled in windows of POOL_WIN tokens ("slots",
# ~one clause each) - i.e. the whole document as ONE stream of ~(document tokens / POOL_WIN) slot vectors.
# Pooling: N_DOC_TOK learned queries (shifted by the mean state of the current chunk) attend over ALL slots of the document
# (multi-head attention, softmax over the whole stream; keys carry a Fourier embedding of the slot's relative position in the
# document and an embedding of its chunk distance to the current chunk); a "nothing" slot is always attendable. The pooled vectors
# become the soft document tokens. Flat (single-level) pooling: no structure other than position is imposed.
POOL_NFREQ = 8


def ctx_settings():
    return {"kind": "docpool", "n_tok": N_DOC_TOK, "win": POOL_WIN, "dim": POOL_DIM, "heads": POOL_HEADS, "ff": POOL_FF,
            "dropout": POOL_DROPOUT, "slot_drop": POOL_SLOT_DROP, "exclude_own": POOL_EXCLUDE_OWN,
            "max_dist": POOL_MAX_DIST, "gate_init": CTX_GATE_INIT}


class DocPool(nn.Module):
    def __init__(self, d_model, s):
        super().__init__()
        self.s = dict(s)
        H, M = s["dim"], s["n_tok"]
        self.in_norm = nn.LayerNorm(d_model)
        self.slot_in = nn.Linear(d_model, H)
        self.pos_mlp = nn.Sequential(nn.Linear(2 * POOL_NFREQ, H), nn.GELU(), nn.Linear(H, H))
        self.dist_emb = nn.Embedding(2 * s["max_dist"] + 1, H)         # signed chunk distance to the current chunk
        nn.init.zeros_(self.dist_emb.weight)
        self.q_base = nn.Parameter(torch.randn(M, H) * 0.02)
        self.q_cond = nn.Linear(d_model, H, bias=False)                  # queries depend on the chunk being summarised
        self.null = nn.Parameter(torch.zeros(1, H))                      # the always-attendable "nothing" slot
        self.read = Readout(H, s["heads"], s["ff"], s["dropout"])
        self.head = PrefixHead(H, d_model, M, s["gate_init"])
        self.register_buffer("freqs", (2.0 ** torch.arange(POOL_NFREQ).float()) * (2.0 * math.pi), persistent=False)

    def forward(self, ctx):
        s = self.s
        f, valid, c, p, k = ctx["f"].float(), ctx["m"], ctx["c"], ctx["p"], ctx["k"]
        B, S, _ = f.shape
        own = valid & (c == k[:, None])                                                 # slots of the chunk being summarised
        use = valid & ~own if s["exclude_own"] else valid
        if self.training and s["slot_drop"] > 0:
            use = use & (torch.rand(B, S, device=f.device) >= s["slot_drop"])
        ow = own.unsqueeze(-1).to(f.dtype)
        own_mean = (f * ow).sum(1) / ow.sum(1).clamp_min(1.0)                          # (B, D); zeros if the chunk has no slot
        ang = p.unsqueeze(-1) * self.freqs                                              # (B, S, NFREQ)
        pos = self.pos_mlp(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))
        dist = (c - k[:, None]).clamp(min=-s["max_dist"], max=s["max_dist"]) + s["max_dist"]
        x = self.slot_in(self.in_norm(f)) + pos + self.dist_emb(dist)
        kv = torch.cat([self.null[None].expand(B, 1, x.size(-1)).to(x.dtype), x], dim=1)               # (B, S+1, H)
        allow = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=f.device), use], dim=1)[:, None, :]  # (B, 1, S+1)
        q = self.q_base[None] + self.q_cond(own_mean)[:, None, :]                        # (B, M, H)
        return self.head(self.read(q, kv, allow))


def build_ctx_module(d_model, s):
    return DocPool(d_model, s)


# ---------------------------- document entries ----------------------------
def pool_states(h, ln):
    """h (b, T, D) float32 token states of b chunks, ln (b,) token counts -> per chunk (slots (s, D) float16, tokens per slot (s,))"""
    win = POOL_WIN
    b, T, D = h.shape
    Tp = -(-T // win) * win
    valid = torch.arange(T, device=h.device)[None, :] < ln[:, None]
    hv = h * valid.unsqueeze(-1).to(h.dtype)
    if Tp > T:
        hv = torch.cat([hv, torch.zeros(b, Tp - T, D, dtype=h.dtype, device=h.device)], dim=1)
        valid = torch.cat([valid, torch.zeros(b, Tp - T, dtype=torch.bool, device=h.device)], dim=1)
    cnt = valid.view(b, Tp // win, win).sum(2)
    slots = hv.view(b, Tp // win, win, D).sum(2) / cnt.clamp_min(1).unsqueeze(-1).to(h.dtype)
    slots, cnt, lens = slots.cpu().numpy(), cnt.cpu().numpy(), ln.cpu().tolist()
    return [(slots[r, :-(-lens[r] // win)].astype(np.float16), cnt[r, :-(-lens[r] // win)].astype(np.float32)) for r in range(b)]


def finish_entry(res, lens):
    """per-chunk slots -> {f: slot features (S, D) fp16, c: chunk index (S,), p: relative position of the slot in the document (S,), n}"""
    total = float(max(sum(lens), 1))
    f, c, p, off = [], [], [], 0
    for j, (sl, cnt) in enumerate(res):
        starts = off + np.arange(len(sl)) * POOL_WIN
        p.append(((starts + cnt / 2.0) / total).astype(np.float32))
        f.append(sl)
        c.append(np.full(len(sl), j, dtype=np.int16))
        off += lens[j]
    return {"f": np.concatenate(f), "c": np.concatenate(c), "p": np.concatenate(p), "n": len(res)}


def make_ctx(items, device, docs=None, blank=False):
    """items: [(document index, chunk index)] -> padded tensors for DocPool. blank=True: an EMPTY document (ablation: no context)"""
    docs = STATE["docs"] if docs is None else docs
    ents = [docs[di] for di, _ in items]
    B, D = len(ents), ents[0]["f"].shape[1]
    S = 1 if blank else max(len(e["c"]) for e in ents)
    f = np.zeros((B, S, D), np.float16)
    c = np.full((B, S), -1, np.int64)
    p = np.zeros((B, S), np.float32)
    m = np.zeros((B, S), bool)
    if not blank:
        for i, e in enumerate(ents):
            n = len(e["c"])
            f[i, :n], c[i, :n], p[i, :n], m[i, :n] = e["f"], e["c"], e["p"], True
    k = np.asarray([0 if blank else min(int(kk), e["n"] - 1) for (_, kk), e in zip(items, ents)], dtype=np.int64)
    return {n: torch.from_numpy(a).to(device) for n, a in (("f", f), ("c", c), ("p", p), ("m", m), ("k", k))}


def doc_n_chunks(docs, di):
    return docs[di]["n"]


def probe_docs(d_model):
    """two synthetic documents (3 chunks / 1 chunk) for the start-up checks"""
    r = np.random.default_rng(7)
    mk = lambda c: {"f": r.normal(size=(len(c), d_model)).astype(np.float16), "c": np.asarray(c, dtype=np.int16),
                    "p": np.linspace(0.05, 0.95, len(c)).astype(np.float32), "n": int(max(c)) + 1}
    return [mk([0, 0, 1, 1, 1, 2, 2]), mk([0, 0])]


@torch.no_grad()
def ctx_profile(model, samples, device, n_items=64):
    """where in the document does the learned pooling look?  (mean attention mass per document-position decile)"""
    mod = model.doc_ctx
    was = mod.training
    mod.eval()
    mod.read.attn.keep = True
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(samples), min(n_items, len(samples)), replace=False)
    mass, null, cnt = np.zeros(10), 0.0, 0
    for i in range(0, len(pick), 8):
        batch = [samples[j] for j in pick[i:i + 8]]
        ctx = batch_ctx(batch, device)
        with amp_ctx():
            mod(ctx)
        att = mod.read.attn.att.float().mean(1).mean(1).cpu().numpy()                  # (B, S+1) over heads and queries
        pp, mm = ctx["p"].cpu().numpy(), ctx["m"].cpu().numpy()
        for r in range(len(batch)):
            null += float(att[r, 0])
            sel = mm[r]
            np.add.at(mass, np.minimum((pp[r][sel] * 10).astype(int), 9), att[r, 1:][sel])
        cnt += len(batch)
    mod.read.attn.keep = False
    mod.train(was)
    return {"attention_mass_by_document_decile": [round(float(x), 4) for x in mass / max(cnt, 1)],
            "attention_on_null_slot": round(null / max(cnt, 1), 4), "gate": round(mod.head.gate.item(), 4)}


# =====================================================================================
# LoRA (own implementation: the backbone is a custom nn.Module, no extra dependency)
# =====================================================================================
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scaling


def lora_settings():
    return {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
            "targets": list(LORA_TARGETS), "train_norms": TRAIN_NORMS, "ctx": ctx_settings()}


def apply_lora(model, s):
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("doc_ctx.")      # Phase-1 weights frozen; the document-context module stays trainable
    n = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in s["targets"]:
                setattr(module, name, LoRALinear(child, s["r"], s["alpha"], s["dropout"]))
                n += 1
    if s.get("train_norms"):
        for m in model.modules():
            if isinstance(m, RMSNorm):
                m.weight.requires_grad = True
    dev = next(model.parameters()).device
    model.to(dev)
    total, trainable = count_params(model), count_params(model, trainable_only=True)
    log(f"LoRA applied to {n} linear layers (r={s['r']}, alpha={s['alpha']}); trainable = LoRA + document-context module: "
        f"{trainable / 1e6:.2f}M trainable / {total / 1e6:.1f}M total ({100 * trainable / total:.2f}%)")


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def save_adapter(path, model, extra):
    atomic_torch_save({"lora": trainable_state(model), "lora_cfg": lora_settings(), **extra}, path)


def load_adapter_state(model, state):
    _, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected adapter keys: {unexpected[:3]}"


@torch.no_grad()
def merge_lora(model):
    """fold W + scaling * B @ A into the base weights (faster decoding, deployable like the Phase-1 model)"""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                w = child.base.weight
                w += (child.scaling * (child.lora_B.float() @ child.lora_A.float())).to(w.dtype)
                setattr(module, name, child.base)
    for p in model.parameters():
        p.requires_grad = False
    log("LoRA merged into the base weights")


# =====================================================================================
# CHUNKING + SUMMARY->CHUNK ALIGNMENT
# =====================================================================================
def split_sentences(text):
    text = " ".join(str(text).split())
    out = []
    for s in SENT_RE.split(text):
        w = s.split()
        for i in range(0, len(w), MAX_SENT_WORDS):
            piece = " ".join(w[i:i + MAX_SENT_WORDS])
            if piece:
                out.append(piece)
    return out


def chunk_document(tok, text):
    """sentence-aligned chunks of <= CHUNK_TOKENS tokens covering the WHOLE document"""
    sents = split_sentences(text)
    if not sents:
        return [" ".join(str(text).split()) or "."]
    budget = chunk_limit() - 16                # margin: separate sentence tokenisation can differ by a token
    lens = [len(x) for x in tok.encode_batch(sents)]
    if max(lens) > budget:                     # a single run-on sentence would be truncated -> cut it by words
        pieces = []
        for s, n in zip(sents, lens):
            if n <= budget:
                pieces.append(s)
                continue
            w = s.split()
            step = max(5, int(len(w) * budget / n * 0.9))
            pieces += [" ".join(w[i:i + step]) for i in range(0, len(w), step)]
        sents = pieces
        lens = [len(x) for x in tok.encode_batch(sents)]
    chunks, cur, cur_len = [], [], 0
    for s, n in zip(sents, lens):
        if cur and cur_len + n > budget:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += n
    if cur:
        chunks.append(cur)
    return [" ".join(c) for c in chunks]


@functools.lru_cache(maxsize=4096)
def _tag_ids(tok_id, k, n):
    return tuple(STATE["tok"].encode(f"[Part {k + 1}/{n}]\n"))


def build_prompt(tok, k, n, chunk_ids):
    STATE["tok"] = tok
    tag = list(_tag_ids(id(tok), k, n)) if USE_PART_TAG else []
    prompt = [tok.unk_id] * N_DOC_TOK + tag + list(chunk_ids)[:chunk_limit()] + STATE["sep_ids"]     # first N_DOC_TOK = placeholders
    return np.asarray(prompt, dtype=np.uint16)


def align_sentences(sents, chunk_texts):
    """chunk index for every summary sentence: IDF-weighted word recall against each chunk
    (IDF over the chunks of this document); near-ties go to the chunk closest to the previous sentence's"""
    n = len(chunk_texts)
    if n == 1 or not sents:
        return [0] * len(sents)
    csets = [set(WORD_RE.findall(c.lower())) for c in chunk_texts]
    df = Counter()
    for cs in csets:
        df.update(cs)
    idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
    out, prev = [], 0
    for si, s in enumerate(sents):
        words = {w for w in WORD_RE.findall(s.lower()) if w in idf}
        denom = sum(idf[w] for w in words)
        scores = [sum(idf[w] for w in words if w in cs) / denom for cs in csets] if denom > 0 else None
        if scores is None or max(scores) < MIN_ALIGN_SCORE:
            c = min(n - 1, int(si * n / len(sents)))
        else:
            best = max(scores)
            c = min((k for k, v in enumerate(scores) if v >= best - ALIGN_TIE_EPS), key=lambda k: (abs(k - prev), k))
        out.append(c)
        prev = c
    return out


def build_samples(tok, recs, rng, keep_empty_p, name):
    st = STATE
    samples, stats = [], Counter()
    pos_acc, pos_cnt = np.zeros(4), np.zeros(4)
    budget = TGT_MAX_TOKENS - len(st["end_ids"])
    t0 = time.time()
    for di, r in enumerate(recs):
        chunk_texts = chunk_document(tok, r["document"])
        n = len(chunk_texts)
        sents = split_sentences(r["summary"])
        assign = align_sentences(sents, chunk_texts)
        per_chunk = [[] for _ in range(n)]
        for s, c in zip(sents, assign):
            per_chunk[c].append(s)
        if n > 1 and len(sents) >= 4:
            for si, c in enumerate(assign):
                q = min(3, int(4 * si / len(sents)))
                pos_acc[q] += c / (n - 1)
                pos_cnt[q] += 1
        chunk_ids = tok.encode_batch(chunk_texts)
        doc_idx = register_doc(encode_doc(STATE["enc"], chunk_ids))        # document features (frozen Phase-1 model)
        tgt_ids = tok.encode_batch([" ".join(x) for x in per_chunk])
        for k in range(n):
            stats["chunks"] += 1
            if not per_chunk[k]:
                stats["empty"] += 1
                if rng.random() > keep_empty_p:
                    continue
                ids = []
            else:
                ids = tgt_ids[k]
                if len(ids) > budget:
                    stats["truncated_targets"] += 1
                    ss = per_chunk[k]
                    while len(ids) > budget and len(ss) > 1:
                        ss = ss[:-1]
                        ids = tok.encode(" ".join(ss))
                    ids = ids[:budget]
            prompt = build_prompt(tok, k, n, chunk_ids[k])
            target = np.asarray(list(ids) + st["end_ids"], dtype=np.uint16)
            samples.append((prompt, target, doc_idx, k))
            stats["prompt_tokens"] += len(prompt)
            stats["target_tokens"] += len(target)
        if (di + 1) % 500 == 0:
            log(f"  [{name}] {di + 1}/{len(recs)} docs | {len(samples):,} samples | {fmt_time(time.time() - t0)}")
    ns = max(len(samples), 1)
    quart = (pos_acc / np.maximum(pos_cnt, 1)).round(2).tolist()
    log(f"[{name}] docs {len(recs):,} | chunks {stats['chunks']:,} | samples {len(samples):,} | "
        f"empty-target chunks {100 * stats['empty'] / max(stats['chunks'], 1):.1f}% | "
        f"truncated targets {stats['truncated_targets']} | avg prompt {stats['prompt_tokens'] / ns:.0f} tok | "
        f"avg target {stats['target_tokens'] / ns:.0f} tok")
    log(f"[{name}] mean aligned chunk position (0=start, 1=end of doc) for summary quartiles 1-4: {quart} "
        f"(rising values = summary follows document order)")
    return samples


def prepare_samples(tok, train_recs, val_recs):
    sig = {"chunk": CHUNK_TOKENS, "tgt": TGT_MAX_TOKENS, "tag": USE_PART_TAG, "sep": SEP_TEXT, "end": END_TEXT,
           "keep_empty": KEEP_EMPTY_P, "sent": MAX_SENT_WORDS, "min_align": MIN_ALIGN_SCORE,
           "n_train": len(train_recs), "n_val": len(val_recs), "seed": SEED, "vocab": tok.vocab_size,
           "first_id": str(train_recs[0]["id"]) if train_recs else "",
           "ctx": {"kind": "docpool", "win": POOL_WIN}, "n_doc_tok": N_DOC_TOK, "chunk_eff": chunk_limit(), "enc": enc_fingerprint()}
    # this session's cache, then the cache of an earlier session attached as an input (saves the chunking time on a resume).
    # The build is deterministic (seeded), so a rebuilt cache is identical - which a mid-epoch resume relies on.
    name = os.path.basename(CACHE_PATH)
    for path in [CACHE_PATH] + [os.path.join(d, name) for d in walk_find(name, [INPUT_ROOT])]:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("sig") == sig:
                log(f"loaded cached chunk samples ({path}): train {len(blob['train']):,} val {len(blob['val']):,}")
                STATE["docs"] = blob["docs"]
                return blob["train"], blob["val"]
            log(f"chunk cache {path}: signature differs -> ignored")
        except Exception as e:
            log(f"chunk cache {path} unreadable ({e}) -> ignored")
    rng = random.Random(SEED)
    STATE["docs"] = []
    train_s = build_samples(tok, train_recs, rng, KEEP_EMPTY_P, "train")
    val_s = build_samples(tok, val_recs, rng, KEEP_EMPTY_P, "val")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH + ".tmp", "wb") as f:
        pickle.dump({"sig": sig, "train": train_s, "val": val_s, "docs": STATE["docs"]}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(CACHE_PATH + ".tmp", CACHE_PATH)
    return train_s, val_s


# =====================================================================================
# TRAINING
# =====================================================================================
def collate(batch, pad_id):
    """right-padded (causal attention => padding never influences earlier positions); labels only on the summary"""
    L = max(len(p) + len(t) for p, t, *_ in batch)
    ids = np.full((len(batch), L), pad_id, dtype=np.int64)
    lab = np.full((len(batch), L), -100, dtype=np.int64)
    for i, (p, t, *_) in enumerate(batch):
        n, m = len(p), len(t)
        ids[i, :n] = p
        ids[i, n:n + m] = t
        lab[i, n:n + m] = t
    return torch.from_numpy(ids[:, :-1]), torch.from_numpy(lab[:, 1:])


def summary_loss_sum(model, x, y, ctx):
    with amp_ctx():
        prefix = model.doc_ctx(ctx)                      # (B, N_DOC_TOK, D) document tokens from the learned context module
        h = model.features(x, prefix)
        sel = y != -100
        logits = F.linear(h[sel], model.embed.weight)
    return F.cross_entropy(logits.float(), y[sel], reduction="sum"), int(sel.sum().item())


def other_doc_sample(samples, j):
    """index of a sample of a DIFFERENT document (deterministic) - for the 'context of another document' ablation"""
    n = len(samples)
    for t in range(1, n):
        j2 = (j + 17 * t) % n
        if samples[j2][2] != samples[j][2]:
            return j2
    return j


@torch.no_grad()
def eval_loss(model, samples, device, mode="real"):
    """validation loss on the summary tokens.  mode: real = the document's own context | other = the context of another document
    (same chunk index) | none = an empty document (the context module gets nothing)"""
    model.eval()
    order = np.argsort([-(len(sm[0]) + len(sm[1])) for sm in samples])
    tot, cnt = 0.0, 0
    for i in range(0, len(order), EVAL_BS):
        sel = order[i:i + EVAL_BS]
        batch = [samples[j] for j in sel]
        x, y = collate(batch, model.cfg.pad_id)
        if mode == "other":
            ctx = make_ctx([(samples[other_doc_sample(samples, j)][2], samples[j][3]) for j in sel], device)
        elif mode == "none":
            ctx = make_ctx([(b[2], b[3]) for b in batch], device, blank=True)
        else:
            ctx = batch_ctx(batch, device)
        loss, n = summary_loss_sum(model, x.to(device), y.to(device), ctx)
        tot += loss.item()
        cnt += n
    return tot / max(cnt, 1)


def make_optimizer(model):
    """LoRA matrices at LR, the (new) document-context module at CTX_LR_MULT x LR; matrices decay, vectors / scalars do not"""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = []
    for is_ctx, mult in ((False, 1.0), (True, CTX_LR_MULT)):
        ps = [p for n, p in named if n.startswith("doc_ctx.") == is_ctx]
        groups += [{"params": [p for p in ps if p.ndim >= 2], "weight_decay": WEIGHT_DECAY, "lr_mult": mult},
                   {"params": [p for p in ps if p.ndim < 2], "weight_decay": 0.0, "lr_mult": mult}]
    groups = [dict(g, lr=LR * g["lr_mult"], base_lr=LR * g["lr_mult"]) for g in groups if g["params"]]
    fused = next(model.parameters()).is_cuda
    return torch.optim.AdamW(groups, lr=LR, betas=(0.9, 0.999), fused=fused)


class EMA:
    """exponential moving average of the trainable (LoRA) weights; `applied()` temporarily swaps it into the model"""

    def __init__(self, model, decay):
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.names = [n for n, _ in named]
        self.params = [p for _, p in named]
        self.shadow = [p.detach().clone().float() for p in self.params]
        self.decay = decay

    @torch.no_grad()
    def update(self, step):
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        torch._foreach_mul_(self.shadow, d)
        torch._foreach_add_(self.shadow, [p.detach().float() for p in self.params], alpha=1.0 - d)

    @contextlib.contextmanager
    def applied(self):
        backup = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for p, s in zip(self.params, self.shadow):
                p.copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(self.params, backup):
                    p.copy_(b)

    def state_dict(self):
        return {n: s.detach().cpu().clone() for n, s in zip(self.names, self.shadow)}

    def load_state_dict(self, sd):
        for n, s in zip(self.names, self.shadow):
            s.copy_(sd[n].to(s.device))


# ---------------------------- learning-rate cycle ----------------------------
def lr_mult(step, prog, sc):
    """linear warm-up (by steps since the start of the LR cycle `sc`) x cosine decay driven by `prog` in [0, 1].
    peak = fraction of the base LR: 1.0 for a fresh run, CONT_LR_PEAK for a continuation."""
    warm = min(1.0, (step - sc["start_step"] + 1) / max(sc["warmup"], 1))
    return sc["peak"] * warm * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def make_schedule(resumed, ck_sched, state, steps_per_epoch):
    """Fresh run: one cycle (warm-up -> cosine over all EPOCHS).
    Resume of a cycle that is still running (session died / time budget reached, same EPOCHS): the same cycle goes on.
    Resume with a different EPOCHS (continuing a finished run): a NEW cycle from the current epoch to EPOCHS."""
    if ck_sched is not None and ck_sched.get("end_epoch") == EPOCHS and ck_sched.get("steps_per_epoch") == steps_per_epoch:
        return dict(ck_sched)
    if resumed and CONT_NEW_LR_CYCLE:
        return {"kind": "continuation", "start_step": state["global_step"], "start_sec": state["train_sec"],
                "start_epoch": state["epoch"], "end_epoch": EPOCHS, "steps_per_epoch": steps_per_epoch,
                "cycle_steps": max((EPOCHS - state["epoch"]) * steps_per_epoch - state["bi"], 1),
                "peak": CONT_LR_PEAK, "warmup": CONT_WARMUP_STEPS}
    return {"kind": "initial", "start_step": 0, "start_sec": 0.0, "start_epoch": 0, "end_epoch": EPOCHS,
            "steps_per_epoch": steps_per_epoch, "cycle_steps": EPOCHS * steps_per_epoch, "peak": 1.0, "warmup": WARMUP_STEPS}


def schedule_progress(state, sched, train_budget):
    step_p = (state["global_step"] - sched["start_step"]) / max(sched["cycle_steps"], 1)
    if not TIME_AWARE_LR:
        return min(1.0, step_p)
    time_p = (state["train_sec"] - sched["start_sec"]) / max(train_budget - sched["start_sec"], 1.0)
    return min(1.0, max(step_p, time_p))


# ---------------------------- checkpoint bookkeeping ----------------------------
def find_previous_latest(roots=None):
    """newest resumable checkpoint of THIS run (this session's folder, then the earlier output attached as an input)"""
    best = None
    for d in walk_find("lora_latest_meta.json", roots or [CKPT_DIR, INPUT_ROOT]):
        meta = read_json(os.path.join(d, "lora_latest_meta.json"))
        if not meta or meta.get("tag") != RUN_TAG or not os.path.exists(os.path.join(d, "lora_latest.pt")):
            continue
        if best is None or meta.get("global_step", -1) > best[1].get("global_step", -1):
            best = (d, meta)
    return best


def carry_over_best(roots=None):
    """copy the best adapter of an earlier session (attached as an input) into this session's folder if it is better"""
    cur = read_json(f"{CKPT_DIR}/lora_best_meta.json")
    for d in walk_find("lora_best_meta.json", roots or [INPUT_ROOT]):
        meta = read_json(os.path.join(d, "lora_best_meta.json"))
        if not meta or meta.get("tag") != RUN_TAG or not os.path.exists(os.path.join(d, "lora_best.pt")):
            continue
        if cur is None or meta["val_loss"] < cur["val_loss"]:
            os.makedirs(CKPT_DIR, exist_ok=True)
            shutil.copy(os.path.join(d, "lora_best.pt"), f"{CKPT_DIR}/lora_best.pt.tmp")
            os.replace(f"{CKPT_DIR}/lora_best.pt.tmp", f"{CKPT_DIR}/lora_best.pt")
            shutil.copy(os.path.join(d, "lora_best_meta.json"), f"{CKPT_DIR}/lora_best_meta.json")
            cur = meta
    return cur


# ---------------------------- training ----------------------------
def train(model, train_s, val_s, device, t_start, stop_at_step=None):
    """LoRA training with mid-epoch checkpoints. Returns a dict: finished (all EPOCHS done), resumed_from
    ((epoch, batch, global_step) of the checkpoint or None), final_step, best_val.
    stop_at_step: simulate a time-out after this many optimizer steps (used by the smoke test)."""
    best_path, best_meta_path = f"{CKPT_DIR}/lora_best.pt", f"{CKPT_DIR}/lora_best_meta.json"
    latest_path, latest_meta_path = f"{CKPT_DIR}/lora_latest.pt", f"{CKPT_DIR}/lora_latest_meta.json"
    best_meta = carry_over_best()
    best_val = best_meta["val_loss"] if best_meta else float("inf")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(model)
    base_lrs = [g["base_lr"] for g in opt.param_groups]
    scaler = make_grad_scaler(AMP["enabled"] and AMP["dtype"] == torch.float16)
    n = len(train_s)
    steps_per_epoch = max(1, n // EFF_BS)
    total_steps = EPOCHS * steps_per_epoch
    state = {"epoch": 0, "bi": 0, "global_step": 0, "train_sec": 0.0, "ep_loss": 0.0, "ep_n": 0}
    history, baseline, ck_ema, ck_sched, val_check, resumed_from = [], None, None, None, None, None

    prev = find_previous_latest()
    if prev is not None:
        d, _ = prev
        ck = torch.load(os.path.join(d, "lora_latest.pt"), map_location="cpu", weights_only=False)
        if ck.get("lora_cfg") != lora_settings():
            raise ValueError(f"the checkpoint in {d} was trained with LoRA settings {ck.get('lora_cfg')} but the current ones are "
                             f"{lora_settings()} - change RUN_TAG for a new run, or restore the old settings")
        load_adapter_state(model, ck["lora"])
        opt.load_state_dict(ck["opt"])
        for g, b in zip(opt.param_groups, base_lrs):
            g["base_lr"] = b                                   # the CURRENT config LR wins over the stored one
        if ck.get("scaler") and scaler.is_enabled():
            try:
                scaler.load_state_dict(ck["scaler"])
            except Exception as e:                             # e.g. saved on a bf16 GPU, resumed on an fp16 GPU
                log(f"  (grad-scaler state not restored: {e})")
        state.update({k: ck[k] for k in state})
        if ck.get("steps_per_epoch") != steps_per_epoch and state["bi"] > 0:
            log(f"WARNING: steps per epoch changed ({ck.get('steps_per_epoch')} -> {steps_per_epoch}, different chunk samples?) "
                f"-> the interrupted epoch is restarted from its first batch")
            state.update(bi=0, ep_loss=0.0, ep_n=0)
        ck_ema, ck_sched, val_check = ck.get("ema"), ck.get("sched"), ck.get("val_check")
        history, baseline = ck.get("history", []), ck.get("baseline")
        resumed_from = (state["epoch"], state["bi"], state["global_step"])
        log(f"RESUMED from {d}: epoch {state['epoch'] + 1}, batch {state['bi']}, step {state['global_step']}, "
            f"train time so far {fmt_time(state['train_sec'])}")
        del ck
    elif REQUIRE_RESUME:
        seen = [(d, (read_json(os.path.join(d, "lora_latest_meta.json")) or {}).get("tag"))
                for d in walk_find("lora_latest_meta.json", [INPUT_ROOT])]
        raise FileNotFoundError(
            f"REQUIRE_RESUME is on, but no earlier checkpoint with tag '{RUN_TAG}' was found under {INPUT_ROOT}. "
            f"lora_latest_meta.json files seen: {seen or 'none'}. Add the earlier run's output (checkpoints/lora_latest.pt + "
            "lora_latest_meta.json) as an input of this notebook and keep RUN_TAG unchanged - or set REQUIRE_RESUME = False.")

    ema = EMA(model, EMA_DECAY) if USE_EMA else None
    if ema is not None and ck_ema is not None:
        ema.load_state_dict(ck_ema)

    def ema_ctx():
        return ema.applied() if ema is not None else contextlib.nullcontext()

    if prev is None:                                           # reference point for the report: Phase-1 model, LoRA still a no-op
        baseline = eval_loss(model, val_s, device)
        history.append({"epoch": 0, "val_loss": baseline, "val_ppl": math.exp(min(baseline, 20))})
        log(f"validation loss of the Phase-1 model (+ untrained context module) before adaptation: {baseline:.4f} (ppl {math.exp(min(baseline, 20)):.2f})")
    elif RESUME_CHECK:
        if val_check:
            with ema_ctx():
                vl = eval_loss(model, val_s, device)
            diff = abs(vl - val_check["loss"])
            ok = diff <= RESUME_CHECK_TOL
            log(f"resume check: validation loss of the restored weights {vl:.4f} vs {val_check['loss']:.4f} when the checkpoint "
                f"was written (diff {diff:.4f}) -> {'OK' if ok else 'MISMATCH'}")
            if not ok and SMOKE_TEST:
                raise RuntimeError("smoke test: the restored weights do not reproduce the saved validation loss - resume is broken")
        else:
            log("resume check skipped (the checkpoint was written mid-epoch, without a validation pass)")
    model.train()

    sched = make_schedule(prev is not None, ck_sched, state, steps_per_epoch)
    log(f"{n:,} train samples | {steps_per_epoch} steps/epoch (effective batch {EFF_BS} chunks) | {EPOCHS} epochs | {total_steps:,} steps")
    log(f"LR cycle: {sched['kind']} | epochs {sched['start_epoch'] + 1}..{sched['end_epoch']} | peak x{sched['peak']:g} | "
        f"warm-up {sched['warmup']} steps | ~{sched['cycle_steps']:,} steps | time-aware LR: {TIME_AWARE_LR}")
    if prev is not None and state["epoch"] >= EPOCHS:
        log(f"the checkpoint is already at epoch {state['epoch']} >= EPOCHS = {EPOCHS}: nothing left to train "
            f"(raise EPOCHS to continue)")

    # training seconds at which THIS session stops (session limit; with TIME_AWARE_LR also the training-time cap of the LR cycle)
    sess_avail = (TIME_BUDGET_H - FINAL_RESERVE_H) * 3600 - (time.time() - t_start)
    train_budget = state["train_sec"] + max(sess_avail, 0.0)
    if TIME_AWARE_LR:
        cap_h = TRAIN_HOURS if sched["kind"] == "initial" else CONT_TRAIN_HOURS
        train_budget = min(train_budget, sched["start_sec"] + cap_h * 3600)
    log(f"this session may train for {fmt_time(train_budget - state['train_sec'])} more "
        f"(session limit {TIME_BUDGET_H} h, {FINAL_RESERVE_H} h reserved for the final prediction)")

    last = [time.time()]

    def tick():
        now = time.time()
        state["train_sec"] += now - last[0]
        last[0] = now

    def save_latest(vcheck=None):
        atomic_torch_save({"lora": trainable_state(model), "lora_cfg": lora_settings(), "opt": opt.state_dict(),
                           "scaler": scaler.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                           **state, "sched": sched, "steps_per_epoch": steps_per_epoch, "history": history,
                           "baseline": baseline, "val_check": vcheck, "tag": RUN_TAG}, latest_path)
        atomic_json({"tag": RUN_TAG, "epoch": state["epoch"], "bi": state["bi"], "global_step": state["global_step"],
                     "train_sec": state["train_sec"], "steps_per_epoch": steps_per_epoch, "sched": sched,
                     "lora_cfg": lora_settings(), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, latest_meta_path)

    def validate(where):
        """validation loss of the EMA weights; saves lora_best.pt when it is the best so far"""
        nonlocal best_val
        with ema_ctx():
            vl = eval_loss(model, val_s, device)
            improved = vl < best_val
            if improved:
                best_val = vl
                save_adapter(best_path, model, {"val_loss": vl, "epochs_done": state["epoch"], "batch": state["bi"],
                                                "global_step": state["global_step"], "tag": RUN_TAG})
                atomic_json({"tag": RUN_TAG, "val_loss": vl, "where": where, "global_step": state["global_step"],
                             "lora_cfg": lora_settings()}, best_meta_path)
        model.train()
        return vl, improved

    stopped, n_skipped, steps_this_run, run, run_n = False, 0, 0, 0.0, 0
    tokens_seen, t_loop = 0, time.time()
    for epoch in range(state["epoch"], EPOCHS):
        perm = np.random.default_rng(SEED + epoch).permutation(n)       # same order on every resume
        model.train()
        for step in range(state["bi"], steps_per_epoch):
            if state["train_sec"] >= train_budget or (stop_at_step is not None and state["global_step"] >= stop_at_step):
                stopped = True
                break
            mult = lr_mult(state["global_step"], schedule_progress(state, sched, train_budget), sched)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * mult
            idxs = sorted(perm[step * EFF_BS:(step + 1) * EFF_BS].tolist(),
                          key=lambda i: -(len(train_s[i][0]) + len(train_s[i][1])))   # similar lengths per micro-batch
            n_tgt = sum(len(train_s[i][1]) for i in idxs)
            step_loss = 0.0
            for m in range(0, len(idxs), MICRO_BS):
                batch = [train_s[i] for i in idxs[m:m + MICRO_BS]]
                x, y = collate(batch, model.cfg.pad_id)
                loss, _ = summary_loss_sum(model, x.to(device), y.to(device), batch_ctx(batch, device))
                loss = loss / n_tgt                      # token-level mean over the whole effective batch
                scaler.scale(loss).backward()
                step_loss += loss.item()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            if math.isfinite(step_loss) and bool(torch.isfinite(gnorm)):
                scaler.step(opt)
                if ema is not None:
                    ema.update(state["global_step"])
                state["ep_loss"] += step_loss
                state["ep_n"] += 1
                run += step_loss
                run_n += 1
            else:                                        # fp16 overflow (the scaler lowers its scale) or a bad batch
                n_skipped += 1
                if n_skipped <= 5 or n_skipped % 50 == 0:
                    log(f"  WARNING: non-finite loss/gradient at step {state['global_step']} - step skipped ({n_skipped} so far)")
            scaler.update()
            opt.zero_grad(set_to_none=True)
            state["global_step"] += 1
            state["epoch"], state["bi"] = epoch, step + 1
            steps_this_run += 1
            tokens_seen += n_tgt
            tick()

            if steps_this_run % LOG_EVERY == 0:
                el = time.time() - t_loop
                sps = el / steps_this_run
                log(f"[ep {epoch + 1}/{EPOCHS}] step {step + 1}/{steps_per_epoch} (global {state['global_step']}) | "
                    f"loss {run / max(run_n, 1):.4f} | gnorm {float(gnorm):.2f} | lr x{mult:.3f} | {sps:.2f}s/step | "
                    f"epoch ETA {fmt_time((steps_per_epoch - step - 1) * sps)} | all-epochs ETA "
                    f"{fmt_time((total_steps - state['global_step']) * sps)} | train time {fmt_time(state['train_sec'])}")
                run, run_n = 0.0, 0
            if state["global_step"] % SAVE_EVERY == 0 and step + 1 < steps_per_epoch:
                save_latest()
                tick()

        if stopped:
            reason = "simulated time-out (smoke test)" if (stop_at_step is not None and state["global_step"] >= stop_at_step) \
                else "training-time budget of this session used up"
            vl, improved = validate(f"stop@epoch{epoch + 1}.{state['bi']}")
            save_latest({"loss": vl, "global_step": state["global_step"]})
            tick()
            log(f"  >> {reason}: stopped inside epoch {epoch + 1} after batch {state['bi']}/{steps_per_epoch} "
                f"(global step {state['global_step']}); val loss {vl:.4f}{' (new best)' if improved else ''}; checkpoint saved. "
                f"Attach this run's output to the next session and run again to continue.")
            break

        # ---- end of epoch ----
        tr = state["ep_loss"] / max(state["ep_n"], 1)
        state.update(epoch=epoch + 1, bi=0, ep_loss=0.0, ep_n=0)
        vl, improved = validate(f"epoch{epoch + 1}")
        history.append({"epoch": epoch + 1, "train_loss": tr, "val_loss": vl, "val_ppl": math.exp(min(vl, 20)),
                        "global_step": state["global_step"], "train_hours": state["train_sec"] / 3600, "best": improved})
        atomic_json(history, f"{CKPT_DIR}/train_history.json")
        save_latest({"loss": vl, "global_step": state["global_step"]})
        tick()
        log(f"===== epoch {epoch + 1}/{EPOCHS} finished | train loss {tr:.4f} | val loss {vl:.4f} (ppl {math.exp(min(vl, 20)):.2f}) | "
            f"{'NEW BEST, saved lora_best.pt' if improved else 'best %.4f' % best_val} | train time {fmt_time(state['train_sec'])}")
        model.train()

    if n_skipped:
        log(f"note: {n_skipped} optimizer step(s) skipped because of a non-finite loss/gradient")
    if not os.path.exists(best_path):                          # e.g. the earlier output was attached without its best adapter
        with ema_ctx():
            save_adapter(best_path, model, {"val_loss": float("nan"), "epochs_done": state["epoch"], "batch": state["bi"],
                                            "global_step": state["global_step"], "tag": RUN_TAG})
    finished = state["epoch"] >= EPOCHS
    log(f"training {'finished' if finished else 'NOT finished'} | epochs done {state['epoch']}/{EPOCHS} | best val loss {best_val:.4f} | "
        f"train time {fmt_time(state['train_sec'])} | session elapsed {fmt_time(time.time() - t_start)}")
    return {"finished": finished, "resumed_from": resumed_from, "final_step": state["global_step"], "best_val": best_val,
            "steps_per_epoch": steps_per_epoch}


# =====================================================================================
# DECODING (batched, KV cache, left-padded prompts; stops at the end marker)
# =====================================================================================
def make_prompt_batch(seqs, pad_id, device):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros(len(seqs), width, dtype=torch.bool)
    for r, s in enumerate(seqs):
        ids[r, width - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[r, width - len(s):] = True
    return ids.to(device), mask.to(device)


def start_generation(model, prompt, prompt_mask, max_new, prefix=None):
    B, W = prompt.shape
    dev = prompt.device
    lengths = prompt_mask.sum(1)
    pos_ids = (torch.arange(W, device=dev)[None] - (W - lengths)[:, None]).clamp(min=0)
    causal = torch.tril(torch.ones(W, W, dtype=torch.bool, device=dev))
    eye = torch.eye(W, dtype=torch.bool, device=dev)
    mask = (causal[None, None] & prompt_mask[:, None, None, :]) | eye[None, None]
    valid = torch.cat([prompt_mask, torch.ones(B, max_new, dtype=torch.bool, device=dev)], dim=1)
    cdtype = AMP["dtype"] if AMP["enabled"] else torch.float32
    cache = model.init_cache(B, W + max_new, cdtype, dev)
    logits = model.forward_cached(prompt, pos_ids, cache, 0, mask, prefix=prefix, pre_off=W - lengths)   # document tokens: first m real positions
    return cache, logits, lengths, valid


def next_logits(model, tok, t, W, cache, lengths, valid):
    pos = (lengths + t)[:, None]
    mask = valid[:, None, None, :W + t + 1]
    return model.forward_cached(tok[:, None], pos, cache, W + t, mask)


def block_ngrams(logits, seq, t, n, pad_id):
    """ban every token that would complete an n-gram already present in seq[:, :t] (all rows at once)"""
    if n < 2 or t < n:
        return
    s = seq[:, :t]
    ctx = s[:, t - n + 1:]
    wins = s.unfold(1, n - 1, 1)[:, : t - n + 1]
    match = (wins == ctx[:, None, :]).all(-1)
    nxt = s[:, n - 1:]
    banned = torch.where(match, nxt, torch.full_like(nxt, pad_id))     # pad is banned anyway -> harmless filler
    logits.scatter_(1, banned, float("-inf"))


@torch.no_grad()
def generate_batch(model, prompts, items, max_new=None, no_repeat=None, rep_penalty=None, docs=None):
    """items: [(document index, chunk index)] per prompt (-> document tokens). Greedy decoding of one summary per prompt; returns [(token_ids_before_stop, hit_length_limit), ...]"""
    max_new = max_new or GEN_MAX_NEW
    no_repeat = NO_REPEAT_NGRAM if no_repeat is None else no_repeat
    rep_penalty = REP_PENALTY if rep_penalty is None else rep_penalty
    model.eval()
    cfg = model.cfg
    dev = next(model.parameters()).device
    ids, mask = make_prompt_batch([p.tolist() for p in prompts], cfg.pad_id, dev)
    B, W = ids.shape
    assert W + max_new <= cfg.max_len, f"prompt {W} + max_new {max_new} exceeds the RoPE table ({cfg.max_len})"
    stop_np = STATE["stop_mask_np"]
    stop_mask = torch.from_numpy(stop_np).to(dev)
    seq = torch.full((B, max_new), cfg.pad_id, dtype=torch.long, device=dev)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    ctx = make_ctx(items, dev, docs)
    with amp_ctx():
        prefix = model.doc_ctx(ctx)
        cache, logits, lengths, valid = start_generation(model, ids, mask, max_new, prefix)
        for t in range(max_new):
            logits = logits.float()
            logits[:, cfg.pad_id] = float("-inf")
            logits[:, cfg.unk_id] = float("-inf")
            if rep_penalty != 1.0 and t > 0:
                prev = seq[:, :t]
                sc = logits.gather(1, prev)
                sc = torch.where(sc > 0, sc / rep_penalty, sc * rep_penalty)
                logits.scatter_(1, prev, sc)
            block_ngrams(logits, seq, t, no_repeat, cfg.pad_id)
            tok = logits.argmax(-1)
            tok = torch.where(finished, torch.full_like(tok, cfg.pad_id), tok)
            seq[:, t] = tok
            finished = finished | stop_mask[tok]
            if bool(finished.all()):
                break
            if t < max_new - 1:
                logits = next_logits(model, tok, t, W, cache, lengths, valid)
    out = []
    for row in seq.cpu().numpy():
        hit = np.flatnonzero(stop_np[row])
        if len(hit):
            out.append((row[:hit[0]].tolist(), False))
        else:
            out.append((row.tolist(), True))
    return out


def gen_with_oom_retry(model, prompts, items):
    try:
        return generate_batch(model, prompts, items)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            raise
        h = len(prompts) // 2
        log(f"  CUDA OOM, splitting batch of {len(prompts)}")
        return gen_with_oom_retry(model, prompts[:h], items[:h]) + gen_with_oom_retry(model, prompts[h:], items[h:])


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def assemble_summary(parts, doc_words):
    """parts: [(chunk_summary_text, hit_length_limit)] in document order -> final summary.
    Drops a cut-off last sentence, near-duplicate sentences and, if needed, trims proportionally to a length cap."""
    kept_sets, per_chunk = [], []
    for text, truncated in parts:
        sents = split_sentences(text)
        if truncated and sents and not END_PUNCT.search(sents[-1]):
            sents = sents[:-1]
        keep = []
        for s in sents:
            ws = set(WORD_RE.findall(s.lower()))
            if len(ws) < 3 or any(jaccard(ws, o) >= DEDUP_JACCARD for o in kept_sets):
                continue
            kept_sets.append(ws)
            keep.append(s)
        per_chunk.append(keep)
    total = sum(len(s.split()) for ks in per_chunk for s in ks)
    cap = MAX_SUMMARY_WORDS
    if MAX_SUMMARY_DOC_RATIO:
        cap = min(cap, max(50, int(MAX_SUMMARY_DOC_RATIO * doc_words)))
    if total > cap > 0:
        trimmed = []
        for ks in per_chunk:
            share = cap * sum(len(s.split()) for s in ks) / total
            out, acc = [], 0
            for s in ks:
                if out and acc >= share:
                    break
                out.append(s)
                acc += len(s.split())
            trimmed.append(out)
        per_chunk = trimmed
    return " ".join(s for ks in per_chunk for s in ks)


def summarize_docs(model, tok, docs, desc="docs"):
    """docs: [(id, text)] -> [summary]; every chunk of every document is decoded, chunks of all documents
    are batched together (sorted by length) for speed"""
    jobs, n_per_doc = [], []
    for di, (_, text) in enumerate(docs):
        chunks = chunk_document(tok, text)
        cids = tok.encode_batch(chunks)
        doc_i = register_doc(encode_doc(STATE["enc"], cids))     # document features (frozen Phase-1 model)
        n_per_doc.append(len(chunks))
        for k in range(len(chunks)):
            jobs.append((di, k, build_prompt(tok, k, len(chunks), cids[k]), doc_i))
    log(f"[{desc}] {len(docs)} documents -> {len(jobs)} chunks (avg {len(jobs) / max(len(docs), 1):.1f} per doc)")
    order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][2]))
    outs, t0 = {}, time.time()
    for s in range(0, len(order), GEN_BS):
        idxs = order[s:s + GEN_BS]
        res = gen_with_oom_retry(model, [jobs[j][2] for j in idxs], [(jobs[j][3], jobs[j][1]) for j in idxs])
        for j, (ids, trunc) in zip(idxs, res):
            outs[(jobs[j][0], jobs[j][1])] = (tok.decode(ids), trunc)
        done = s + len(idxs)
        if (s // GEN_BS) % 5 == 0 or done == len(order):
            el = time.time() - t0
            log(f"  [{desc}] {done}/{len(order)} chunks | {fmt_time(el)} | ETA {fmt_time(el / done * (len(order) - done))}")
    summaries, n_trunc, n_empty = [], 0, 0
    for di, (_, text) in enumerate(docs):
        parts = [outs[(di, k)] for k in range(n_per_doc[di])]
        n_trunc += sum(1 for _, tr in parts if tr)
        n_empty += sum(1 for tx, _ in parts if not tx.strip())
        summ = assemble_summary(parts, len(text.split()))
        if not summ.strip():                              # never submit an empty string
            summ = " ".join(text.split()[:150])
        summaries.append(summ)
    log(f"[{desc}] chunk outputs: {n_empty} empty (skipped chunks), {n_trunc} hit the length limit")
    return summaries


# =====================================================================================
# EVALUATION (ROUGE-1/L F1 offline; BERTScore if the bert-score package + roberta-large are available)
# =====================================================================================
def _words(t):
    return WORD_RE.findall(t.lower())


def rouge1_f1(ref, hyp):
    a, b = Counter(_words(ref)), Counter(_words(hyp))
    ov = sum((a & b).values())
    if ov == 0:
        return 0.0
    p, r = ov / sum(b.values()), ov / sum(a.values())
    return 2 * p * r / (p + r)


def rougeL_f1(ref, hyp, cap=2500):
    a, b = _words(ref)[:cap], _words(hyp)[:cap]
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(b), lcs / len(a)
    return 2 * p * r / (p + r)


def bertscore_f1(hyps, refs, device):
    if not USE_BERTSCORE:
        return None
    try:
        try:
            import bert_score  # noqa: F401
        except Exception:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bert-score"], check=False)
        from bert_score import score as bs_score
        _, _, f = bs_score([h or "empty" for h in hyps], [r or "empty" for r in refs], lang="en",
                           device=str(device), batch_size=16, verbose=False)
        return float(f.mean())
    except Exception as e:
        log(f"  (BERTScore unavailable: {type(e).__name__}: {str(e)[:100]})")
        return None


def eval_generation(model, tok, recs, device, tag):
    hyps = summarize_docs(model, tok, [(r["id"], r["document"]) for r in recs], desc=tag)
    refs = [r["summary"] for r in recs]
    res = {"rouge1": float(np.mean([rouge1_f1(r, h) for r, h in zip(refs, hyps)])),
           "rougeL": float(np.mean([rougeL_f1(r, h) for r, h in zip(refs, hyps)])),
           "bertscore_f1": bertscore_f1(hyps, refs, device),
           "avg_words_hyp": float(np.mean([len(h.split()) for h in hyps])),
           "avg_words_ref": float(np.mean([len(r.split()) for r in refs]))}
    log(f"[{tag}] " + " | ".join(f"{k} {v:.4f}" if v is not None else f"{k} n/a" for k, v in res.items()))
    log(f"[{tag}] sample reference: {refs[0][:400]}")
    log(f"[{tag}] sample generated: {hyps[0][:400]}")
    return res, hyps


# =====================================================================================
# SELF-CHECKS (cheap; they run in every run, before / after the expensive parts)
# =====================================================================================
PROBE_TEXTS = ("The appellant filed a writ petition before the High Court challenging the order of detention passed under the "
               "Act, and the Court held that the detention was valid.",
               "Held: the appeal is allowed.")
PROBE_ITEMS = [(0, 1), (1, 0)]            # (synthetic document, chunk index) of the two probe prompts


def probe_prompts(tok):
    """two prompts of different length (so the batched decoder has to left-pad one of them)"""
    return [build_prompt(tok, 0, 1, tok.encode(t)) for t in PROBE_TEXTS]


@torch.no_grad()
def check_cached_decoding(model, tok, n_new=8):
    """KV-cached batched decoding (left padding, RoPE positions, masks, document tokens at the start of every row) must produce the
    same greedy tokens as plain forward passes over the growing sequence. Runs in fp32 BEFORE training, so a decoding bug is found in
    seconds, not after 9 hours."""
    prompts = probe_prompts(tok)
    dev = next(model.parameters()).device
    docs = probe_docs(model.cfg.d_model)
    was_training, amp_was = model.training, AMP["enabled"]
    AMP["enabled"] = False
    model.eval()
    try:
        got = generate_batch(model, prompts, PROBE_ITEMS, max_new=n_new, no_repeat=0, docs=docs)
        stop = STATE["stop_mask_np"]
        for pi, (p, (g, _)) in enumerate(zip(prompts, got)):
            prefix = model.doc_ctx(make_ctx([PROBE_ITEMS[pi]], dev, docs))
            seq, ref = p.tolist(), []
            for _ in range(n_new):
                lg = model(torch.tensor([seq], device=dev), prefix)[0, -1].float()
                lg[model.cfg.pad_id] = float("-inf")
                lg[model.cfg.unk_id] = float("-inf")
                t = int(lg.argmax())
                if stop[t]:
                    break
                ref.append(t)
                seq.append(t)
            if list(g) != ref:
                raise RuntimeError(f"cached decoding differs from plain forward passes for probe {pi}: {list(g)} vs {ref}")
    finally:
        AMP["enabled"] = amp_was
        model.train(was_training)
    log(f"decoding check OK: cached batched greedy decoding with document tokens == plain forward passes ({n_new} tokens, "
        f"2 prompts of different length)")


@torch.no_grad()
def check_context_wiring(model, tok):
    """(1) the document tokens are really read by the decoder (another document changes the hidden states), (2) the context module
    is batch-invariant (padding of the document in a batch changes nothing), (3) a row padded on the RIGHT inside a training batch
    gives the same hidden states as the row alone."""
    dev = next(model.parameters()).device
    docs = probe_docs(model.cfg.d_model)
    p0, p1 = [p.tolist() for p in probe_prompts(tok)]
    was_training, amp_was = model.training, AMP["enabled"]
    AMP["enabled"] = False
    model.eval()
    try:
        both = model.doc_ctx(make_ctx(PROBE_ITEMS, dev, docs))
        alone = [model.doc_ctx(make_ctx([it], dev, docs)) for it in PROBE_ITEMS]
        d_batch = max((both[i:i + 1] - alone[i]).abs().max().item() for i in range(2))
        L = max(len(p0), len(p1))
        ids = torch.full((2, L), model.cfg.pad_id, dtype=torch.long, device=dev)
        ids[0, :len(p0)] = torch.tensor(p0, device=dev)
        ids[1, :len(p1)] = torch.tensor(p1, device=dev)
        hb = model.features(ids, both)
        h1 = model.features(torch.tensor([p1], device=dev), alone[1])
        d_pad = (hb[1, :len(p1)] - h1[0]).abs().max().item()
        x0 = torch.tensor([p0], device=dev)
        effect = (model.features(x0, alone[0]) - model.features(x0, alone[1])).abs().max().item()
    finally:
        AMP["enabled"] = amp_was
        model.train(was_training)
    log(f"context wiring check: batch invariance of the module {d_batch:.1e} | right-padded row {d_pad:.1e} | effect of the document "
        f"tokens on the hidden states {effect:.2e}")
    if d_batch > 1e-3 or d_pad > 1e-3 or effect < 1e-7:
        raise RuntimeError("document-context wiring is broken (module not batch invariant, padding leaks, or the decoder ignores the "
                           "document tokens)")


@torch.no_grad()
def merge_with_check(model, tok):
    """fold LoRA into the base weights and verify that the network function did not change"""
    dev = next(model.parameters()).device
    docs = probe_docs(model.cfg.d_model)
    xs = [torch.tensor([p.tolist()], device=dev) for p in probe_prompts(tok)]
    model.eval()
    pre = [model.doc_ctx(make_ctx([it], dev, docs)) for it in PROBE_ITEMS]
    before = [model.features(x, pr) for x, pr in zip(xs, pre)]
    merge_lora(model)
    after = [model.features(x, pr) for x, pr in zip(xs, pre)]
    diff = max((a - b).abs().max().item() for a, b in zip(before, after))
    log(f"merge check: max |hidden(LoRA) - hidden(merged)| = {diff:.2e}")
    if diff > 2e-3:
        msg = f"merged model differs from the LoRA model (max diff {diff:.2e})"
        if SMOKE_TEST:
            raise RuntimeError(msg)
        log("  WARNING: " + msg)
    return diff


def ctx_diagnostics(model, val_s, device):
    """does the decoder use the document context?  validation loss with the real context / the context of another document / none"""
    res = {m: eval_loss(model, val_s, device, m) for m in ("real", "other", "none")}
    res["gain_vs_no_context"] = res["none"] - res["real"]
    res["gain_vs_other_document"] = res["other"] - res["real"]
    log(f"CONTEXT ABLATION (validation loss on summary tokens): real document context {res['real']:.4f} | context of ANOTHER document "
        f"{res['other']:.4f} | no context {res['none']:.4f}   (a decoder that uses the context loses when it is wrong or missing)")
    prof = ctx_profile(model, val_s, device)
    log(f"context module profile: {prof}")
    atomic_json({"ablation_val_loss": res, "profile": prof, "ctx_settings": ctx_settings()}, os.path.join(CKPT_DIR, "ctx_report.json"))
    return res


# =====================================================================================
# MAIN
# =====================================================================================
def find_adapter():
    p = os.path.join(CKPT_DIR, "lora_best.pt")
    if os.path.exists(p):
        return p
    hits = [d for d in walk_find("lora_best.pt", [INPUT_ROOT]) if (read_json(os.path.join(d, "lora_best_meta.json")) or {}).get("tag") == RUN_TAG]
    if hits:
        return os.path.join(hits[0], "lora_best.pt")
    raise FileNotFoundError("lora_best.pt not found - train first or attach the earlier run's output")


def write_submission(test_recs, summaries):
    os.makedirs(OUT_DIR, exist_ok=True)
    preds = [{"id": r["id"], "summary": s} for r, s in zip(test_recs, summaries)]   # list format (Option A)
    assert len(preds) == len(test_recs) and all(p["summary"].strip() for p in preds)
    path, zpath = os.path.join(OUT_DIR, "prediction.json"), os.path.join(OUT_DIR, "submission.zip")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname="prediction.json")
    lens = [len(p["summary"].split()) for p in preds]
    log(f"saved {path} and {zpath} | {len(preds)} summaries | words min/avg/max = "
        f"{min(lens)}/{np.mean(lens):.0f}/{max(lens)}")


def smoke_subset(recs, n, key="document"):
    return [r for r in recs if len(str(r[key]).split()) <= SMOKE_MAX_DOC_WORDS][:n]


def main(train_raw=None, test_raw=None):
    t_start = time.time()
    if SMOKE_TEST and os.path.isdir(WORK_DIR):
        shutil.rmtree(WORK_DIR)                         # a smoke run always starts clean (and never touches the real run's folder)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    for d in (CKPT_DIR, CACHE_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        AMP["enabled"] = True
        AMP["dtype"] = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        log(f"GPU: {torch.cuda.get_device_name()} | autocast {AMP['dtype']}")
    else:
        log("WARNING: no GPU visible (Kaggle: Settings -> Accelerator). This will be extremely slow.")
    log(f"run tag {RUN_TAG} | smoke test: {SMOKE_TEST} | epochs {EPOCHS} | effective batch {EFF_BS} chunks | work dir {WORK_DIR}")

    if train_raw is None:
        train_raw, test_raw = load_records("train"), load_records("test")
    model, cfg = load_model_with_ctx(device)
    tok = find_tokenizer(cfg.vocab_size)
    init_state(tok)
    STATE["enc"] = load_encoder(device)                  # frozen copy of the Phase-1 model: it produces the document features
    check_cached_decoding(model, tok)                    # fail fast: before any training time is spent
    check_context_wiring(model, tok)

    recs = clean_train_records(train_raw)
    random.Random(SEED).shuffle(recs)
    if SMOKE_TEST:
        val_recs = smoke_subset(recs, N_SMOKE["val"])
        used = {r["id"] for r in val_recs}
        train_recs = smoke_subset([r for r in recs if r["id"] not in used], N_SMOKE["train"])
    else:
        val_recs, train_recs = recs[:VAL_DOCS], recs[VAL_DOCS:]
    log(f"documents: {len(train_recs):,} train / {len(val_recs)} validation")

    result, val_s = None, None
    if RUN_TRAIN:
        train_s, val_s = prepare_samples(tok, train_recs, val_recs)
        if MAX_TRAIN_CHUNKS and len(train_s) > MAX_TRAIN_CHUNKS:
            train_s = random.Random(SEED).sample(train_s, MAX_TRAIN_CHUNKS)
            log(f"training set capped to {len(train_s):,} chunks")
        apply_lora(model, lora_settings())
        if SMOKE_TEST:
            spe = max(1, len(train_s) // EFF_BS)
            stop = min(SMOKE_STOP_STEP, max(1, spe // 2))
            log(f"\n===== SMOKE: training session 1 - cut after {stop} of {EPOCHS * spe} steps ({spe} steps/epoch) =====")
            r1 = train(model, train_s, val_s, device, t_start, stop_at_step=stop)
            assert not r1["finished"] and r1["final_step"] == stop, f"session 1 should stop at step {stop}: {r1}"
            log("\n===== SMOKE: training session 2 - fresh model, resumes from the checkpoint =====")
            model, cfg = load_model_with_ctx(device)
            apply_lora(model, lora_settings())
            result = train(model, train_s, val_s, device, t_start)
            assert result["resumed_from"] is not None and result["resumed_from"][2] == r1["final_step"], \
                f"session 2 did not resume from session 1's checkpoint: {result['resumed_from']}"
            assert result["finished"] and result["final_step"] == EPOCHS * spe, \
                f"expected {EPOCHS * spe} steps in total, got {result['final_step']}"
            log("SMOKE: interrupted + resumed training OK "
                f"(resumed at epoch {result['resumed_from'][0] + 1}, batch {result['resumed_from'][1]}, step {result['resumed_from'][2]})")
        else:
            result = train(model, train_s, val_s, device, t_start)
        ck = torch.load(os.path.join(CKPT_DIR, "lora_best.pt"), map_location="cpu", weights_only=False)
        load_adapter_state(model, ck["lora"])
        log(f"best adapter restored (val loss {ck['val_loss']:.4f}, after {ck['epochs_done']} epochs + {ck['batch']} batches, "
            f"step {ck['global_step']})")
    else:
        ck = torch.load(find_adapter(), map_location="cpu", weights_only=False)
        attach_doc_ctx(model, ck["lora_cfg"]["ctx"])
        apply_lora(model, ck["lora_cfg"])
        load_adapter_state(model, ck["lora"])

    if val_s:
        try:
            ctx_diagnostics(model, val_s, device)
        except Exception as e:                           # a diagnostic must never cost the submission of a long run
            if SMOKE_TEST:
                raise
            log(f"  WARNING: context diagnostics failed ({type(e).__name__}: {str(e)[:150]})")

    merge_with_check(model, tok)
    model.eval()
    if SAVE_MERGED:
        atomic_torch_save({"model": model.state_dict(), "config": asdict(cfg)}, os.path.join(CKPT_DIR, "merged_model.pt"))

    if RUN_VAL_GEN and val_recs:
        res, _ = eval_generation(model, tok, val_recs[:VAL_GEN_DOCS], device, "val")
        atomic_json(res, os.path.join(CKPT_DIR, "val_generation_metrics.json"))

    if RUN_PREDICT and test_raw is not None:
        test_recs = [{"id": r["id"], "document": " ".join(str(r["document"]).split())} for r in test_raw]
        if SMOKE_TEST:
            test_recs = smoke_subset(test_recs, N_SMOKE["test"])
        summaries = summarize_docs(model, tok, [(r["id"], r["document"]) for r in test_recs], desc="test")
        write_submission(test_recs, summaries)
        if SMOKE_TEST:
            preds = read_json(os.path.join(OUT_DIR, "prediction.json"))
            assert preds and len(preds) == len(test_recs) and all(p["summary"].strip() for p in preds)

    if SMOKE_TEST:
        log("\n" + "=" * 90 + "\nSMOKE TEST PASSED: decoding + context-wiring checks, document-feature cache, interrupted + resumed training, "
            "best adapter, context ablation, LoRA merge, validation generation, test prediction and submission files all work.\n"
            "Now set SMOKE_TEST = False and run for real.\n" + "=" * 90)
    elif result is not None and not result["finished"]:
        log("\nTRAINING IS NOT FINISHED (session time budget). A submission was written from the best adapter so far.\n"
            "To continue: Save & Commit this notebook, attach its output as an input of the next session (same RUN_TAG, same other "
            "inputs), run again - it resumes at the saved epoch/batch.")
    log(f"done in {fmt_time(time.time() - t_start)}")


if __name__ == "__main__":
    main()
