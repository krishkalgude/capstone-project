"""
CS781 Phase 6 - Abstractive legal summarization
LoRA fine-tuning of the Phase-1 legal decoder LM (RoPE + GQA + SwiGLU, 32k byte-level BPE)
with document chunking: EVERY chunk of EVERY document is summarised, summaries are stitched in order.

Why chunking is mandatory here
    docs   : ~4.4k words on average (up to 140k)      -> far beyond the 1024-token context
    summary: ~840 words on average (~1.2k tokens)     -> longer than the context as well
So a document-level (document -> summary) sample cannot even fit. Instead:

TRAIN  (reference summaries exist per document only, not per chunk)
  1. split each document into sentence-aligned chunks of <= CHUNK_TOKENS tokens (no text is dropped)
  2. split the reference summary into sentences, align every sentence to the chunk that covers it
     best (IDF-weighted word recall, ties resolved towards the previous sentence's chunk)
     -> chunk-level pseudo targets
  3. sample = "[Part k/n]\\n<chunk>\\n\\nSUMMARY:\\n<aligned summary sentences>\\n\\n"
     loss only on the summary part. Chunks with nothing to say get an empty target
     (KEEP_EMPTY_P of them are kept) so the model learns to skip unimportant chunks.
INFER
  every chunk is summarised (greedy + no-repeat n-gram), outputs are concatenated in document order,
  near-duplicate sentences are removed, optional length cap.

Only LoRA matrices (and optionally RMSNorm gains) are trained; the Phase-1 weights stay frozen.
No external pretrained weights / APIs are used (bert-score / roberta is used ONLY to report a
validation number, never for training or decoding).

Kaggle setup
    * attach the dataset holding the Phase-1 Model-2 checkpoint (best.pt, best_meta.json)
    * attach the dataset holding the BPE output (tokenizer.json, meta.json)
    * Settings -> Accelerator: GPU (T4 / P100), Internet: ON (Phase-6 data is pulled from Hugging Face)
    * HF token: Kaggle secret named HF (or env HF_TOKEN)
    * a Kaggle GPU session lasts at most 9 h  ->  TIME_BUDGET_H = 9 (1 h is kept for validation + test prediction)
Run as a script or paste into a single cell. Everything lands in /kaggle/working/<RUN_TAG>/
(checkpoints/, cache/, submission/).

1) SMOKE TEST FIRST (~10-15 min):  SMOKE_TEST = True
   Tiny subset of the REAL data (40 train / 8 val / 4 test docs), 2 epochs, and the whole pipeline end to end, including
        - a check that KV-cached batched decoding gives the same tokens as plain forward passes (runs before EVERY training)
        - training session 1, cut in the middle of epoch 1 (simulated time-out)  ->  checkpoint
        - training session 2 = fresh model that RESUMES from that checkpoint: resume check, epoch bookkeeping, best-adapter saving
        - LoRA merge check, validation generation (ROUGE), test prediction, prediction.json + submission.zip
   It prints SMOKE TEST PASSED at the end. A smoke run uses its own folder (<RUN_TAG>_smoke) and never touches the real run.

2) REAL RUN:  SMOKE_TEST = False, EPOCHS = 10 (TOTAL). Checkpoints (~50 MB):
        lora_latest.pt / lora_latest_meta.json  every SAVE_EVERY optimizer steps (mid-epoch) and at every epoch end -> resume
        lora_best.pt   / lora_best_meta.json    best validation loss so far (EMA weights), used for the prediction
        train_history.json                      per-epoch train / validation loss (for the report)
   If the session ends first (time budget, crash, disconnect): Save & Commit, then in the NEXT session attach that notebook's
   output as an input and run again with the same RUN_TAG and the same inputs. The log must show
   "RESUMED ... epoch k, batch b" and a "resume check" line. A submission is written at the end of EVERY session
   (from the best adapter so far), also when training is not finished yet.

3) CONTINUING a finished run (e.g. 5 -> 10 epochs): attach the earlier output, raise EPOCHS to the new TOTAL. Training goes on from
   the checkpoint's epoch as a NEW LR cycle (re-warm-up to CONT_LR_PEAK x LR, cosine down at EPOCHS).
"""
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

RUN_TAG = "p6_lora_chunked_v1" + ("_smoke" if SMOKE_TEST else "")
IS_KAGGLE = os.path.isdir("/kaggle/working")
INPUT_ROOT = "/kaggle/input"
WORK_DIR = ("/kaggle/working" if IS_KAGGLE else ".") + f"/{RUN_TAG}"
CKPT_DIR = f"{WORK_DIR}/checkpoints"
CACHE_DIR = f"{WORK_DIR}/cache"
CACHE_PATH = f"{CACHE_DIR}/prepared_chunks_v1.pkl"
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
    STATE.update(sep_ids=sep, end_ids=end, stop_list=stop, stop_mask_np=mask)
    max_prompt = CONTEXT_LENGTH - TGT_MAX_TOKENS
    assert CHUNK_TOKENS + 24 + len(sep) <= max_prompt, "CHUNK_TOKENS + TGT_MAX_TOKENS do not fit in the context"
    log(f"separator {len(sep)} tokens | end marker {end} | {len(stop)} stop tokens | "
        f"prompt <= {CHUNK_TOKENS + 24 + len(sep)}, target <= {TGT_MAX_TOKENS}, context {CONTEXT_LENGTH}")


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

    def _rope(self, T, pos_ids=None):
        if pos_ids is None:
            return self.rope_cos[:T][None, None], self.rope_sin[:T][None, None]
        return self.rope_cos[pos_ids][:, None], self.rope_sin[pos_ids][:, None]

    def features(self, idx):
        """final hidden states (B, T, D) - lets the loss be computed on summary positions only"""
        cos, sin = self._rope(idx.size(1))
        x = self.drop(self.embed(idx))
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.final_norm(x)

    def forward(self, idx):
        return F.linear(self.features(idx), self.embed.weight)

    def init_cache(self, B, total_len, dtype, device):
        kv, hd = self.cfg.n_kv_heads, self.cfg.d_model // self.cfg.n_heads
        return [{"k": torch.zeros(B, kv, total_len, hd, dtype=dtype, device=device),
                 "v": torch.zeros(B, kv, total_len, hd, dtype=dtype, device=device)} for _ in self.blocks]

    def forward_cached(self, idx, pos_ids, cache, start, attn_mask):
        cos, sin = self._rope(idx.size(1), pos_ids)
        x = self.embed(idx)
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
            "targets": list(LORA_TARGETS), "train_norms": TRAIN_NORMS}


def apply_lora(model, s):
    for p in model.parameters():
        p.requires_grad = False
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
    log(f"LoRA applied to {n} linear layers (r={s['r']}, alpha={s['alpha']}): "
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
    budget = CHUNK_TOKENS - 16                 # margin: separate sentence tokenisation can differ by a token
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
    prompt = tag + list(chunk_ids)[:CHUNK_TOKENS] + STATE["sep_ids"]
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
            samples.append((prompt, target))
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
           "first_id": str(train_recs[0]["id"]) if train_recs else ""}
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
                return blob["train"], blob["val"]
            log(f"chunk cache {path}: signature differs -> ignored")
        except Exception as e:
            log(f"chunk cache {path} unreadable ({e}) -> ignored")
    rng = random.Random(SEED)
    train_s = build_samples(tok, train_recs, rng, KEEP_EMPTY_P, "train")
    val_s = build_samples(tok, val_recs, rng, KEEP_EMPTY_P, "val")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH + ".tmp", "wb") as f:
        pickle.dump({"sig": sig, "train": train_s, "val": val_s}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(CACHE_PATH + ".tmp", CACHE_PATH)
    return train_s, val_s


# =====================================================================================
# TRAINING
# =====================================================================================
def collate(batch, pad_id):
    """right-padded (causal attention => padding never influences earlier positions); labels only on the summary"""
    L = max(len(p) + len(t) for p, t in batch)
    ids = np.full((len(batch), L), pad_id, dtype=np.int64)
    lab = np.full((len(batch), L), -100, dtype=np.int64)
    for i, (p, t) in enumerate(batch):
        n, m = len(p), len(t)
        ids[i, :n] = p
        ids[i, n:n + m] = t
        lab[i, n:n + m] = t
    return torch.from_numpy(ids[:, :-1]), torch.from_numpy(lab[:, 1:])


def summary_loss_sum(model, x, y):
    with amp_ctx():
        h = model.features(x)
        sel = y != -100
        logits = F.linear(h[sel], model.embed.weight)
    return F.cross_entropy(logits.float(), y[sel], reduction="sum"), int(sel.sum().item())


@torch.no_grad()
def eval_loss(model, samples, device):
    model.eval()
    order = np.argsort([-(len(p) + len(t)) for p, t in samples])
    tot, cnt = 0.0, 0
    for i in range(0, len(order), EVAL_BS):
        x, y = collate([samples[j] for j in order[i:i + EVAL_BS]], model.cfg.pad_id)
        loss, n = summary_loss_sum(model, x.to(device), y.to(device))
        tot += loss.item()
        cnt += n
    return tot / max(cnt, 1)


def make_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p.ndim >= 2], "weight_decay": WEIGHT_DECAY},
              {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}]
    groups = [dict(g, lr=LR, base_lr=LR) for g in groups if g["params"]]
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
        log(f"validation loss of the Phase-1 model before adaptation: {baseline:.4f} (ppl {math.exp(min(baseline, 20)):.2f})")
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
                x, y = collate([train_s[i] for i in idxs[m:m + MICRO_BS]], model.cfg.pad_id)
                loss, _ = summary_loss_sum(model, x.to(device), y.to(device))
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


def start_generation(model, prompt, prompt_mask, max_new):
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
    logits = model.forward_cached(prompt, pos_ids, cache, 0, mask)
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
def generate_batch(model, prompts, max_new=None, no_repeat=None, rep_penalty=None):
    """greedy decoding of one summary per prompt; returns [(token_ids_before_stop, hit_length_limit), ...]"""
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
    with amp_ctx():
        cache, logits, lengths, valid = start_generation(model, ids, mask, max_new)
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


def gen_with_oom_retry(model, prompts):
    try:
        return generate_batch(model, prompts)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            raise
        h = len(prompts) // 2
        log(f"  CUDA OOM, splitting batch of {len(prompts)}")
        return gen_with_oom_retry(model, prompts[:h]) + gen_with_oom_retry(model, prompts[h:])


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
        n_per_doc.append(len(chunks))
        for k in range(len(chunks)):
            jobs.append((di, k, build_prompt(tok, k, len(chunks), cids[k])))
    log(f"[{desc}] {len(docs)} documents -> {len(jobs)} chunks (avg {len(jobs) / max(len(docs), 1):.1f} per doc)")
    order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][2]))
    outs, t0 = {}, time.time()
    for s in range(0, len(order), GEN_BS):
        idxs = order[s:s + GEN_BS]
        res = gen_with_oom_retry(model, [jobs[j][2] for j in idxs])
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


def probe_prompts(tok):
    """two prompts of different length (so the batched decoder has to left-pad one of them)"""
    return [build_prompt(tok, 0, 1, tok.encode(t)) for t in PROBE_TEXTS]


@torch.no_grad()
def check_cached_decoding(model, tok, n_new=8):
    """KV-cached batched decoding (left padding, RoPE positions, masks) must produce the same greedy tokens as plain forward
    passes over the growing sequence. Runs in fp32 BEFORE training, so a decoding bug is found in seconds, not after 9 hours."""
    prompts = probe_prompts(tok)
    dev = next(model.parameters()).device
    was_training, amp_was = model.training, AMP["enabled"]
    AMP["enabled"] = False
    model.eval()
    try:
        got = generate_batch(model, prompts, max_new=n_new, no_repeat=0)
        stop = STATE["stop_mask_np"]
        for pi, (p, (g, _)) in enumerate(zip(prompts, got)):
            seq, ref = p.tolist(), []
            for _ in range(n_new):
                lg = model(torch.tensor([seq], device=dev))[0, -1].float()
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
    log(f"decoding check OK: cached batched greedy decoding == plain forward passes ({n_new} tokens, 2 prompts of different length)")


@torch.no_grad()
def merge_with_check(model, tok):
    """fold LoRA into the base weights and verify that the network function did not change"""
    dev = next(model.parameters()).device
    xs = [torch.tensor([p.tolist()], device=dev) for p in probe_prompts(tok)]
    model.eval()
    before = [model.features(x) for x in xs]
    merge_lora(model)
    after = [model.features(x) for x in xs]
    diff = max((a - b).abs().max().item() for a, b in zip(before, after))
    log(f"merge check: max |hidden(LoRA) - hidden(merged)| = {diff:.2e}")
    if diff > 2e-3:
        msg = f"merged model differs from the LoRA model (max diff {diff:.2e})"
        if SMOKE_TEST:
            raise RuntimeError(msg)
        log("  WARNING: " + msg)
    return diff


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
    model, cfg = load_phase1_model(device)
    tok = find_tokenizer(cfg.vocab_size)
    init_state(tok)
    check_cached_decoding(model, tok)                    # fail fast: before any training time is spent

    recs = clean_train_records(train_raw)
    random.Random(SEED).shuffle(recs)
    if SMOKE_TEST:
        val_recs = smoke_subset(recs, N_SMOKE["val"])
        used = {r["id"] for r in val_recs}
        train_recs = smoke_subset([r for r in recs if r["id"] not in used], N_SMOKE["train"])
    else:
        val_recs, train_recs = recs[:VAL_DOCS], recs[VAL_DOCS:]
    log(f"documents: {len(train_recs):,} train / {len(val_recs)} validation")

    result = None
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
            model, cfg = load_phase1_model(device)
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
        apply_lora(model, ck["lora_cfg"])
        load_adapter_state(model, ck["lora"])

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
        log("\n" + "=" * 90 + "\nSMOKE TEST PASSED: decoding check, interrupted + resumed training, best adapter, LoRA merge, "
            "validation generation, test prediction and submission files all work.\nNow set SMOKE_TEST = False and run for real.\n"
            + "=" * 90)
    elif result is not None and not result["finished"]:
        log("\nTRAINING IS NOT FINISHED (session time budget). A submission was written from the best adapter so far.\n"
            "To continue: Save & Commit this notebook, attach its output as an input of the next session (same RUN_TAG, same other "
            "inputs), run again - it resumes at the saved epoch/batch.")
    log(f"done in {fmt_time(time.time() - t_start)}")


if __name__ == "__main__":
    main()
