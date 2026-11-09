"""
CS781 Phase 6 - Abstractive legal summarization
KNOWLEDGE DISTILLATION  teacher: Model 3 (RoPE + GQA + SwiGLU + MTP)  ->  student: Model 4 (MSM transformer) fine-tuned with LoRA.
Both models were pretrained on the SAME 50k word-level BPE (vocab_50k.json + merges_50k.json), so teacher and student predict the
very same 50,000 symbols and their output distributions can be compared token by token.  Document chunking as before: EVERY chunk
of EVERY document is summarised, the chunk summaries are stitched in document order.

Student loss on every summary token (loss only on the summary part of "[part k/n] <chunk> summary of the above : <summary> <EOS>")
        L = (1 - KD_ALPHA) * CE(student, reference token)  +  KD_ALPHA * KD_TEMP^2 * KL( softmax(teacher/T) || softmax(student/T) )
    CE   = ordinary cross-entropy against the aligned reference summary sentences ("hard" labels)
    KL   = response-based distillation of the teacher's full next-token distribution over the 50,000 tokens ("soft" labels, Hinton
           et al. 2015); the T^2 factor keeps the gradient scale independent of the temperature
    KD_ALPHA = 0 switches the teacher off completely: the same script then is the plain-LoRA BASELINE for Model 4 (own RUN_TAG),
    which is what the distilled student has to be compared with.

Teacher = the Phase-6 FINE-TUNED Model 3: the Phase-1 best.pt of Model 3 + the LoRMA adapter (lorma_best.pt) trained by
    phase6_lorma_model3.py, folded into the weights.  A teacher that was never fine-tuned to summarise only knows how to continue
    the document, so by default the script refuses to start without a fine-tuned teacher (ALLOW_PRETRAINED_TEACHER = False).
    The teacher is frozen, evaluated in eval mode, and queried online: one extra forward pass per micro-batch (no logits are stored).

Student = Phase-1 Model 4 (3 x [Local-128 -> Local-256 -> Memory-Global], d_model 512, 8 q / 2 kv heads, 32 learned memory
    tokens) + LoRA (own implementation, merged after training):  on q/k/v/o and the SwiGLU matrices of every block and on the
    memory WRITE path (wq/wk/wv/write_out) and gate of the Memory-Global layers.  The learned latent memory (mem_init) and the final
    RMSNorm gain are trained too (at BASE_LR_MULT x LR), as in the Phase-5 distillation notebook.

Model-3 / Model-4 specifics handled here
    * both models were pretrained on a tokenizer that lower-cases the text, has NO whitespace / newline tokens and - through the
      `\\a-z` range in its WORD_PATTERN - cannot represent ASCII punctuation at all -> the end-of-summary marker is the EOS token.
      EOS never occurs in the pretraining windows of either model, so the student gets a small trainable correction of its EOS
      output row (TRAIN_EOS_ROW).  A fine-tuned teacher has learned EOS during its own Phase-6 run (its EOS row is trained), so its
      distribution is used at the EOS positions as well; a pretrained-only teacher is not asked about EOS (CE only there, and the
      EOS column is left out of its KL term).
    * generated text is re-capitalised (word forms learnt from the training summaries) and every chunk summary is closed with a
      full stop before it is scored / submitted
    * only head 1 (next token) of the multi-token-prediction models is used (both for the teacher's targets and for decoding)
    * the tokenizer is verified with BOTH pretrained models (LM loss on Phase-6 text vs. each checkpoint's validation loss)
    * Model 4's memory layers are position-based (chunk = position // 128): training uses the same positions as decoding, and the
      start-up checks prove causality across chunk borders and KV-cached decoding == plain forward passes
No external pretrained weights / APIs are used (bert-score / roberta is used ONLY to report a validation number,
never for training or decoding).

Kaggle setup
    * attach the dataset holding the Phase-1 checkpoints of Model 4 AND Model 3 (best.pt + best_meta.json of each)
    * attach the output of the Phase-6 Model-3 LoRMA run (lorma_best.pt + lorma_best_meta.json) = the fine-tuned teacher
    * attach the dataset holding the tokenizer (vocab_50k.json, merges_50k.json, meta_50k.json)
    * Settings -> Accelerator: GPU (T4 / P100), Internet: ON (Phase-6 data is pulled from Hugging Face)
    * HF token: Kaggle secret named HF (or env HF_TOKEN)
    * a Kaggle GPU session lasts at most 9 h  ->  TIME_BUDGET_H = 9 (1 h is kept for validation + test prediction)
Run as a script or paste into a single cell. Everything lands in /kaggle/working/<RUN_TAG>/
(checkpoints/, cache/, submission/).

1) SMOKE TEST FIRST (~10-15 min):  SMOKE_TEST = True
   Tiny subset of the REAL data (40 train / 8 val / 4 test docs), 2 epochs, and the whole pipeline end to end, including
        - tokenizer selection with both pretrained LMs, the teacher's adapter merge, the KD-loss self-check
        - the block-local attention / causality / KV-cached decoding checks of the student
        - a check that a fresh LoRA adapter is the exact identity (network output unchanged)
        - training session 1, cut in the middle of epoch 1 (simulated time-out)  ->  checkpoint
        - training session 2 = fresh student that RESUMES from that checkpoint: resume check, epoch bookkeeping, best adapter
        - LoRA merge check, validation generation (ROUGE), test prediction, prediction.json + submission.zip
   It prints SMOKE TEST PASSED at the end. A smoke run uses its own folder (<RUN_TAG>_smoke) and never touches the real run.
   (In the smoke test a missing fine-tuned teacher is tolerated, so the plumbing can be checked before the teacher exists.)

2) REAL RUN:  SMOKE_TEST = False, EPOCHS = 10 (TOTAL). Checkpoints (~40 MB):
        student_latest.pt / student_latest_meta.json   every SAVE_EVERY optimizer steps (mid-epoch) and at every epoch end -> resume
        student_best.pt   / student_best_meta.json     best validation CE so far (EMA weights), used for the prediction
        train_history.json                             per-epoch train loss / validation CE / KD term / teacher agreement
        kd_report.json                                 teacher, alpha, T, validation CE of teacher and student (for the report)
   If the session ends first (time budget, crash, disconnect): Save & Commit, then in the NEXT session attach that notebook's
   output as an input and run again with the same RUN_TAG and the same inputs. The log must show
   "RESUMED ... epoch k, batch b" and a "resume check" line. A submission is written at the end of EVERY session
   (from the best adapter so far), also when training is not finished yet.

3) CONTINUING a finished run (e.g. 5 -> 10 epochs): attach the earlier output, raise EPOCHS to the new TOTAL. Training goes on from
   the checkpoint's epoch as a NEW LR cycle (re-warm-up to CONT_LR_PEAK x LR, cosine down at EPOCHS).

4) BASELINE FOR THE COMPARISON: set KD_ALPHA = 0 (new RUN_TAG automatically, no teacher is loaded) and repeat 1) - 2). The
   difference between the two runs (ROUGE / BERTScore on the validation documents, validation CE) is the effect of distillation.
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
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================================
# CONFIG
# =====================================================================================
# ------------------------------------------------------------------
# SWITCH: run once with True (tiny subset, 2 epochs, simulated interruption + resume, ~10-15 min) to verify the whole
# pipeline end to end, then set False for the real run.
# ------------------------------------------------------------------
SMOKE_TEST = False

# ------------------------------------------------------------------
# EXPERIMENT SWITCH: knowledge distillation on / off.  Each setting has its own RUN_TAG / folder, so both can be run.
#   student loss = (1 - KD_ALPHA) * CE(hard summary tokens) + KD_ALPHA * KD_TEMP^2 * KL(teacher || student)
#   KD_ALPHA = 0  ->  no teacher is loaded: plain LoRA fine-tuning of Model 4 = the baseline of the distillation experiment
# ------------------------------------------------------------------
KD_ALPHA = 0.5                 # weight of the soft (teacher) term; 0 = off
KD_TEMP = 2.0                  # softmax temperature of the soft term
USE_TEACHER = KD_ALPHA > 0
assert 0.0 <= KD_ALPHA <= 1.0 and KD_TEMP > 0

RUN_TAG = ("p6_kd_m3_to_m4_lora_v1" if USE_TEACHER else "p6_lora_m4_baseline_v1") + ("_smoke" if SMOKE_TEST else "")
IS_KAGGLE = os.path.isdir("/kaggle/working")
INPUT_ROOT = "/kaggle/input"
WORK_DIR = ("/kaggle/working" if IS_KAGGLE else ".") + f"/{RUN_TAG}"
CKPT_DIR = f"{WORK_DIR}/checkpoints"
CACHE_DIR = f"{WORK_DIR}/cache"
CACHE_PATH = f"{CACHE_DIR}/prepared_chunks_m3_v1.pkl"      # same file name as the Model-3 LoRMA run: its cache is reused if attached
OUT_DIR = f"{WORK_DIR}/submission"

# ---- Phase-1 artefacts (None = auto-search /kaggle/input) ----
STUDENT_CKPT = None           # path to best.pt of Model 4 (None = search /kaggle/input for the tag below)
STUDENT_TAG = "decoder_only_msm_rope_gqa_swiglu_mtp3_v2"      # MODEL_TAG of the Model-4 notebook
TEACHER_CKPT = None           # path to best.pt of Model 3 (None = search /kaggle/input for the tag below)
TEACHER_TAG = "decoder_only_rope_gqa_swiglu_mtp4_v1"          # MODEL_TAG of the Model-3 notebook
PHASE1_TOKENIZER_DIR = None   # folder with vocab_50k.json + merges_50k.json (+ meta_50k.json), or tokenizer.json
TOKENIZER_CHECK = True        # score every tokenizer candidate with the pretrained LMs and refuse a mismatching one
TOKENIZER_STRICT = True       # True: raise when even the best candidate is far above a checkpoint's validation loss
TOKENIZER_MARGIN = 3.0        # nats above a checkpoint's validation loss that are still accepted (wrong tokenizer: ~8+ nats)

# ---- the teacher: Phase-1 Model 3 + its Phase-6 LoRMA adapter ----
TEACHER_ADAPTER = None        # path to lorma_best.pt of the fine-tuned Model-3 teacher (None = search /kaggle/input)
TEACHER_RUN_PREFIX = "p6_lorma_m3_"   # RUN_TAG prefix of phase6_lorma_model3.py (variants "plus" / "pi"); the best one is used
ALLOW_PRETRAINED_TEACHER = False      # True: with no fine-tuned Model 3 available, distil from the Phase-1 Model 3 as it is
                                      #       (it only continues text; a weak teacher - use for the plumbing test only)
TEACHER_STRICT = True         # raise when the rebuilt teacher does not reproduce the validation loss stored in its adapter
TEACHER_CHECK_TOL = 0.05      # tolerated difference (nats). A failing check means: wrong tokenizer / changed chunking constants

# ---- data ----
HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
PHASE6_FILES = {"train": "Phase_6_Summ/Train/train.json", "test": "Phase_6_Summ/Test/test.json"}
VAL_DOCS = 150                 # held-out documents (no public dev split exists)
MIN_SUMMARY_WORDS = 20         # drop empty / degenerate reference summaries
MAX_TRAIN_SUMMARY_DOC_RATIO = 0.9   # drop pairs whose summary is almost as long as the document

# ---- chunking / alignment (identical to the Model-3 LoRMA run, so that the teacher's validation loss is reproducible) ----
CONTEXT_LENGTH = 1024          # the context both Phase-1 models were trained with
CHUNK_TOKENS = 560             # source tokens per chunk
TGT_MAX_TOKENS = 420           # max summary tokens per chunk (incl. the end marker)
MAX_SENT_WORDS = 200           # run-on "sentences" are cut into pieces of at most this many words
MIN_ALIGN_SCORE = 0.10         # below this, a summary sentence is placed by relative position instead
ALIGN_TIE_EPS = 0.02           # chunks within this score of the best are ties -> pick closest to previous
KEEP_EMPTY_P = 0.30            # share of "nothing to summarise" chunks kept as (chunk -> empty) samples
USE_PART_TAG = True            # prefix "[part k/n]" so the model knows where in the document it is
SEP_TEXT = "summary of the above :"   # the tokenizer lower-cases and has no newline / whitespace tokens
END_TEXT = "<|endoftext|>"     # label only: the end-of-summary marker IS the EOS token (no newline token exists)
MAX_TRAIN_CHUNKS = None        # e.g. 30000 to cap the training set for a quick run

# ---- student LoRA ----
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05           # on the adapter branch input
# attention (local blocks and the READ path of the memory blocks) and SwiGLU
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "w_gate", "w_up", "w_down")
# memory WRITE path (what the memory tokens collect from their chunk) and the update gate of the Memory-Global layers
LORA_MEM_TARGETS = ("wq_proj", "wk_proj", "wv_proj", "write_out", "gate_proj")
ADAPT_MEMORY = True
TRAIN_MEM_INIT = True          # also train the learned latent memory (32 x 512) ...
TRAIN_FINAL_NORM = True        # ... and the gain of the final RMSNorm; both at BASE_LR_MULT x LR (Phase-5 distillation did the same)
BASE_LR_MULT = 0.1
TRAIN_NORMS = False            # also train the gains of every RMSNorm (tiny)
TRAIN_EOS_ROW = True           # + one trainable 512-dim correction of the EOS output row: EOS never occurs in the Phase-1 windows, so
                               #   its output row was only ever pushed DOWN and the adapters alone cannot steer to it
FT_DROPOUT = 0.1               # FFN dropout of the student while fine-tuning (= Phase-1 setting)

# ---- optimisation ----
EPOCHS = 10                    # TOTAL number of epochs (a resume continues from the checkpoint's epoch up to this number)
MICRO_BS = 4                   # lower to 2 (raise ACCUM) on CUDA OOM - the teacher forward pass and two 50k-wide softmaxes need memory
ACCUM = 8
LR = 4e-4
MIN_LR_RATIO = 0.05
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
USE_EMA = True                 # exponential moving average of the trainable weights; validation + best adapter use the EMA weights
EMA_DECAY = 0.998
EVAL_BS = 8
LOG_EVERY = 25
SAVE_EVERY = 100               # optimizer steps between mid-epoch checkpoints (student_latest.pt is only ~40 MB)
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
GEN_WINDOW_WORDS = 25          # tokenizer without punctuation: de-duplicate / trim in windows of this many words
MAX_SUMMARY_WORDS = 3000       # hard cap on the stitched summary
MAX_SUMMARY_DOC_RATIO = 0.45   # ... and cap relative to document length (reference p95 ~ 0.44)
VAL_GEN_DOCS = 60              # held-out docs used for the end-to-end ROUGE / BERTScore check
SAVE_MERGED = False            # also save the merged (LoRA folded in) student weights

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
    ALLOW_PRETRAINED_TEACHER = True                # the plumbing can be tested before the fine-tuned teacher exists
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
# PHASE-1 TOKENIZER
#   (a) the custom WORD-LEVEL BPE that the Model-3 and Model-4 notebooks load: vocab_50k.json + merges_50k.json (+ meta_50k.json).
#       It lower-cases the text, splits it into words / digit runs / punctuation runs, drops ALL whitespace (there are no
#       newline tokens) and every word ends with "</w>".  Class copied from the Model-3 notebook.
#   (b) optionally the byte-level BPE of legal_bpe_50k.ipynb (tokenizer.json, HF `tokenizers`).
#   A model only works with the tokenizer it was pretrained with, so every candidate is scored with the pretrained models
#   (student AND teacher: language-model loss on Phase-6 text, compared with each checkpoint's own validation loss) and the
#   best one is used.  Both models must have been trained on the same 50k BPE.
# =====================================================================================
WORD_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]+|[^\s\a-zA-Z0-9]+"""
)


class LegalBPE:
    """custom word-level BPE of the Model-3 notebook (identical encode / decode)"""
    kind = "custom word-level BPE"
    needs_detok = True        # lower-cased, whitespace-free output -> detok() + restore_case() on the generated text

    def __init__(self, vocab_path, merges_path, meta_path=None):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(merges_path, "r", encoding="utf-8") as f:
            merges = json.load(f)
        h = hashlib.md5()                                  # tokenizer fingerprint (part of the chunk-cache signature)
        for path in (vocab_path, merges_path, meta_path):
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    h.update(f.read())
        self.fingerprint = h.hexdigest()[:12]
        self.ranks = {(a, b): i for i, (a, b) in enumerate(merges)}      # lower rank = earlier merge
        self.inv = {int(i): t for t, i in self.vocab.items()}
        self.vocab_size = max(self.inv) + 1
        assert len(self.inv) == self.vocab_size, "vocab ids are not contiguous"
        self.pad_id = self.vocab["<|pad|>"]
        self.unk_id = self.vocab["<|unk|>"]
        self.eos_id = self.vocab["<|endoftext|>"]
        self.bos_id = self.eos_id
        self.mask_id = self.eos_id
        self.special_ids = {self.pad_id, self.unk_id, self.eos_id}
        meta = {}
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        for key, val in (("pad_id", self.pad_id), ("unk_id", self.unk_id), ("eos_id", self.eos_id)):
            if key in meta and meta[key] != val:
                raise ValueError(f"meta.json {key}={meta[key]} disagrees with vocab ({val})")
        self.cache = {}

    def _word_ids(self, word):
        hit = self.cache.get(word)
        if hit is not None:
            return hit
        syms = list(word) + ["</w>"]
        while len(syms) > 1:
            best, best_rank = None, None
            for i in range(len(syms) - 1):
                r = self.ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = (syms[i], syms[i + 1]), r
            if best is None:
                break
            merged = best[0] + best[1]
            out, i = [], 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == best[0] and syms[i + 1] == best[1]:
                    out.append(merged)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out
        ids = [self.vocab.get(s, self.unk_id) for s in syms]
        if len(self.cache) > 3_000_000:
            self.cache.clear()
        self.cache[word] = ids
        return ids

    def encode(self, text):
        ids = []
        for w in WORD_PATTERN.findall(text.lower()):
            ids.extend(self._word_ids(w))
        return ids

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def decode(self, ids):
        toks = [self.inv.get(int(i), "") for i in ids if int(i) not in self.special_ids]
        return "".join(toks).replace("</w>", " ").strip()


class HFByteBPE:
    """byte-level BPE of legal_bpe_50k.ipynb (tokenizer.json); case and whitespace are preserved"""
    kind = "byte-level BPE (tokenizer.json)"
    needs_detok = False

    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        v = self.tokenizer.get_vocab()
        self.pad_id, self.unk_id, self.eos_id = v["<|pad|>"], v["<|unk|>"], v["<|endoftext|>"]
        self.bos_id, self.mask_id = v.get("<|bos|>", self.eos_id), v.get("<|mask|>", self.eos_id)
        self.special_ids = {self.pad_id, self.unk_id, self.eos_id, self.bos_id, self.mask_id}
        with open(path, "rb") as f:
            self.fingerprint = hashlib.md5(f.read()).hexdigest()[:12]

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def encode_batch(self, texts):
        return [e.ids for e in self.tokenizer.encode_batch(list(texts))]

    def decode(self, ids):
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=True).replace("\ufffd", "")


def find_tokenizers(real_vocab):
    """every tokenizer under the search roots whose vocabulary size matches the model"""
    roots = [PHASE1_TOKENIZER_DIR] if PHASE1_TOKENIZER_DIR else [INPUT_ROOT, os.getcwd(), "/kaggle/working"]
    found, seen = [], set()
    for d in walk_find("vocab_50k.json", roots):
        v, m = os.path.join(d, "vocab_50k.json"), os.path.join(d, "merges_50k.json")
        if not os.path.exists(m) or d in seen:
            continue
        seen.add(d)
        try:
            tok = LegalBPE(v, m, os.path.join(d, "meta_50k.json"))
        except Exception as e:
            log(f"  (could not load the word-level tokenizer in {d}: {type(e).__name__}: {str(e)[:80]})")
            continue
        if tok.vocab_size == real_vocab:
            found.append((f"{tok.kind} in {d}", tok))
    for d in walk_find("tokenizer.json", roots):
        if d in seen:
            continue
        seen.add(d)
        try:
            tok = HFByteBPE(os.path.join(d, "tokenizer.json"))
        except Exception as e:
            log(f"  (could not load tokenizer.json in {d}: {type(e).__name__}: {str(e)[:80]})")
            continue
        if tok.vocab_size == real_vocab:
            found.append((f"{tok.kind} in {d}", tok))
    return found


@torch.no_grad()
def lm_nll(model, tok, texts, skip=128, n_tokens=384):
    """mean next-token NLL (head 1) of the pretrained model on a stretch of each text. A wrong tokenizer gives ~8-11 nats,
    the right one lands near the validation loss the checkpoint was saved with."""
    dev = next(model.parameters()).device
    was = model.training
    model.eval()
    tot, cnt = 0.0, 0
    for t in texts:
        ids = tok.encode(t[:12000])
        ids = ids[skip:skip + n_tokens + 1] if len(ids) > skip + 64 else ids[:n_tokens + 1]
        if len(ids) < 33:
            continue
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=dev)
        y = torch.tensor(ids[1:], dtype=torch.long, device=dev)
        with amp_ctx():
            lg = model(x)[0]
        tot += F.cross_entropy(lg.float(), y, reduction="sum").item()
        cnt += y.numel()
    model.train(was)
    return tot / max(cnt, 1)


def choose_tokenizer(checks, real_vocab, texts):
    """checks: [(label, pretrained model, its pretraining validation loss)] - EVERY model that is going to be used (student and,
    when distilling, teacher) must reproduce its own validation loss with the chosen tokenizer."""
    cands = find_tokenizers(real_vocab)
    if not cands:
        raise FileNotFoundError(
            f"no tokenizer with {real_vocab} tokens found under {PHASE1_TOKENIZER_DIR or INPUT_ROOT}. Attach the dataset holding "
            "vocab_50k.json + merges_50k.json (+ meta_50k.json) - the tokenizer the Model-3 / Model-4 notebooks load - or "
            "tokenizer.json, or set PHASE1_TOKENIZER_DIR.")
    scored = []
    for name, tok in cands:
        rows, ok = [], True
        for label, model, ref in checks:
            nll = lm_nll(model, tok, texts)
            limit = (ref if math.isfinite(ref) else 4.0) + TOKENIZER_MARGIN
            log(f"tokenizer candidate: {name} | {label}: LM loss on Phase-6 text {nll:.3f} "
                f"(checkpoint validation loss {ref:.3f}, limit {limit:.3f})")
            rows.append(nll)
            ok = ok and nll <= limit
        scored.append((sum(rows), ok, name, tok))
    scored.sort(key=lambda s: s[0])
    _, ok, name, tok = scored[0]
    if TOKENIZER_CHECK and not ok:
        msg = (f"the best tokenizer ({name}) is far above the validation loss of at least one pretrained model (see the lines "
               f"above): it is not the tokenizer the Phase-1 models were pretrained with. Attach vocab_50k.json + merges_50k.json "
               f"(+ meta_50k.json) from the pretraining runs - both models must have been trained on the SAME 50k BPE.")
        if TOKENIZER_STRICT:
            raise RuntimeError(msg)
        log("  WARNING: " + msg)
    for label, model, _ in checks:
        c = model.cfg
        if (tok.pad_id, tok.unk_id, tok.eos_id) != (c.pad_id, c.unk_id, c.eos_id):
            raise RuntimeError(f"special ids of the tokenizer (pad/unk/eos = {tok.pad_id}/{tok.unk_id}/{tok.eos_id}) differ from the "
                               f"{label} config ({c.pad_id}/{c.unk_id}/{c.eos_id})")
    log(f"using tokenizer: {name} (vocab {tok.vocab_size}, pad {tok.pad_id}, unk {tok.unk_id}, eos {tok.eos_id})")
    return tok


def init_state(tok, vocab_rows):
    """separator / end marker / banned ids / stop set. The end marker is the EOS token (the tokenizer has no newline token).
    vocab_rows = number of output columns of the model that decodes (the student: exactly the 50,000 tokens)."""
    sep, end = tok.encode(SEP_TEXT), [tok.eos_id]
    assert sep and tok.unk_id not in sep, "the separator text must be representable by the tokenizer"
    ban = ({tok.pad_id, tok.unk_id, tok.bos_id, tok.mask_id} - {tok.eos_id}) | set(range(tok.vocab_size, vocab_rows))
    mask = np.zeros(vocab_rows, dtype=bool)
    mask[tok.eos_id] = True
    punct = "." in tok.decode(tok.encode("a. b, c"))
    STATE.update(sep_ids=sep, end_ids=end, ban_list=sorted(ban), stop_mask_np=mask, punct=punct)
    if not punct:
        log("NOTE: this tokenizer cannot represent ASCII punctuation (the WORD_PATTERN range `\\a-z` swallows it), so both models were "
            "pretrained on - and the student will generate - text WITHOUT punctuation. ROUGE ignores punctuation; chunk summaries are "
            "closed with a full stop and de-duplicated in windows of GEN_WINDOW_WORDS words instead of sentences.")
    max_prompt = CONTEXT_LENGTH - TGT_MAX_TOKENS
    assert CHUNK_TOKENS + 24 + len(sep) <= max_prompt, "CHUNK_TOKENS + TGT_MAX_TOKENS do not fit in the context"
    log(f"separator {sep} ({len(sep)} tokens) | end marker = EOS id {tok.eos_id} | {len(ban)} banned ids | "
        f"prompt <= {CHUNK_TOKENS + 24 + len(sep)}, target <= {TGT_MAX_TOKENS}, context {CONTEXT_LENGTH}")


# =====================================================================================
# THE TWO PHASE-1 MODELS
#   TEACHER  Model 3 (model3_mtp_final.ipynb): RoPE + GQA + SwiGLU, weight-tied embedding (50,048 rows = 50,000 tokens + padding
#            rows), 8 layers, d_model 640, multi-token-prediction heads.  Classes copied verbatim, prefix M3 (the two models use
#            different RMSNorm / RoPE code, so their classes must not be mixed up).
#   STUDENT  Model 4 (model4.ipynb): MSM transformer - 3 x [Local-128 -> Local-256 -> Memory-Global], d_model 512, tied embedding of
#            exactly 50,000 rows.  Classes copied verbatim (the training-time MTP loss is left out; only head 1 is used here).
#   Both read the output of their head 1 through the tied embedding; the helpers lm_features / lm_logits give the next-token
#   features and the logits over the SAME 50,000 symbols for either model.
# =====================================================================================
def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def lm_features(model, x):
    """normalised next-token (head 1) features of either model: (B, T) ids -> (B, T, D)"""
    if hasattr(model, "trunk"):                                     # Model 4 (student)
        return model.head_input(model.trunk(x), 0)
    return model.head_z(model.hidden(x), 0)                         # Model 3 (teacher)


def lm_logits(model, z):
    """logits over the 50,000 real tokens from features z (..., D); the student adds its EOS-row correction"""
    return model.out_logits(z)


def find_ckpt(tag, override, what):
    """best.pt of the Phase-1 model carrying MODEL_TAG `tag` (lowest validation loss; latest.pt as the last resort)"""
    if override:
        return override
    cands, seen = [], []
    for fname, mname in (("best.pt", "best_meta.json"), ("latest.pt", "latest_meta.json")):
        for d in walk_find(fname, [INPUT_ROOT, os.getcwd(), "/kaggle/working"]):
            meta = read_json(os.path.join(d, mname)) or {}
            seen.append((os.path.join(d, fname), meta.get("tag")))
            if meta.get("tag") == tag:
                if fname == "best.pt":
                    cands.append((0, float(meta.get("val_loss", 1e9)), os.path.join(d, fname)))
                else:
                    cands.append((1, -float(meta.get("global_step", 0)), os.path.join(d, fname)))
    if not cands:                                      # best_meta.json missing? look at the tag stored inside best.pt
        for d in walk_find("best.pt", [INPUT_ROOT, os.getcwd(), "/kaggle/working"]):
            path = os.path.join(d, "best.pt")
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False)
                if ck.get("tag") == tag:
                    cands.append((0, float(ck.get("val_loss", 1e9)), path))
                del ck
            except Exception as e:
                log(f"  (could not inspect {path}: {type(e).__name__}: {str(e)[:80]})")
    if not cands:
        raise FileNotFoundError(f"no Phase-1 checkpoint of the {what} with tag '{tag}' found. Seen: "
                                f"{seen or 'no best.pt / latest.pt at all'}. Attach the pretraining output (best.pt + best_meta.json) "
                                f"or set {'STUDENT_CKPT' if what == 'student' else 'TEACHER_CKPT'}.")
    cands.sort()
    log(f"Phase-1 checkpoint of the {what}: {cands[0][2]}")
    return cands[0][2]


# ------------------------------- TEACHER: Model 3 (verbatim, prefix M3) -------------------------------
@dataclass
class M3Config:
    vocab_size: int = 50048         # embedding rows (50,000 real tokens rounded up to a multiple of 64)
    real_vocab: int = 50000         # ids >= real_vocab never occur
    d_model: int = 640
    n_layers: int = 8
    n_heads: int = 10
    n_kv_heads: int = 2
    ffn_hidden: int = 1536
    dropout: float = 0.1
    max_len: int = 1088
    mtp_heads: int = 3
    mtp_weights: list = field(default_factory=lambda: [1.0, 0.5, 0.25])
    pad_id: int = 0
    unk_id: int = 1
    eos_id: int = 2
    bos_id: int = 3
    mask_id: int = 4


class M3RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.eps).to(x.dtype)
        return x * inv_rms * self.weight.to(x.dtype)

def m3_build_rope(head_dim, max_len, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len).float(), inv_freq)
    return freqs.cos(), freqs.sin()          # each (max_len, head_dim/2)


def m3_apply_rope(x, cos, sin):
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)

    x1, x2 = x.chunk(2, dim=-1)

    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

def m3_repeat_kv(x, groups):
    # (B, KV, T, hd) -> (B, KV*groups, T, hd); query head h reads kv head h // groups
    if groups == 1:
        return x
    B, KV, T, hd = x.shape
    return x[:, :, None].expand(B, KV, groups, T, hd).reshape(B, KV * groups, T, hd)


class M3Attention(nn.Module):
    """causal grouped-query self-attention with RoPE (10 query heads share 2 key/value heads)"""

    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.n_heads
        self.kv = cfg.n_kv_heads
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
        q = m3_apply_rope(q, cos, sin)
        k = m3_apply_rope(k, cos, sin)
        k, v = m3_repeat_kv(k, self.groups), m3_repeat_kv(v, self.groups)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))

    def forward_cached(self, x, cos, sin, k_cache, v_cache, start, attn_mask):
        """T new tokens go into cache slots [start, start+T); they attend to slots [0, start+T)"""
        B, T, _ = x.shape
        q, k, v = self._qkv(x)
        q = m3_apply_rope(q, cos, sin)
        k = m3_apply_rope(k, cos, sin)
        k_cache[:, :, start:start + T] = k.to(k_cache.dtype)
        v_cache[:, :, start:start + T] = v.to(v_cache.dtype)
        keys = m3_repeat_kv(k_cache[:, :, :start + T], self.groups).to(q.dtype)
        vals = m3_repeat_kv(v_cache[:, :, :start + T], self.groups).to(q.dtype)
        out = F.scaled_dot_product_attention(q, keys, vals, attn_mask=attn_mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class M3SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class M3Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = M3RMSNorm(cfg.d_model)
        self.attn = M3Attention(cfg)
        self.norm2 = M3RMSNorm(cfg.d_model)
        self.ffn = M3SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

    def forward_cached(self, x, cos, sin, cache, start, attn_mask):
        x = x + self.attn.forward_cached(self.norm1(x), cos, sin, cache["k"], cache["v"], start, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class M3MTPHead(nn.Module):
    """residual projection + M3RMSNorm:  z = M3RMSNorm(x + W x).  The logits are z @ E^T with the shared tied embedding E.
    (x is the raw residual stream of the last block; this M3RMSNorm plays the role of the usual final norm.)"""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = M3RMSNorm(dim)

    def forward(self, x):
        return self.norm(x + self.proj(x))


class M3DecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)       # also the (tied) output projection of every head
        self.blocks = nn.ModuleList([M3Block(cfg) for _ in range(cfg.n_layers)])
        self.heads = nn.ModuleList([M3MTPHead(cfg.d_model) for _ in range(cfg.mtp_heads)])
        self.drop = nn.Dropout(cfg.dropout)
        cos, sin = m3_build_rope(cfg.d_model // cfg.n_heads, cfg.max_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # damp the residual-branch output projections (GPT-2 style)
        scale = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)
        # every head starts as the identity branch, i.e. all four heads begin identical and then specialise
        for h in self.heads:
            nn.init.zeros_(h.proj.weight)
        self._ban = None
        self.register_parameter("eos_delta", None)     # (Phase 6) filled when a fine-tuned LoRMA adapter with an EOS row is attached

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope(self, T, pos_ids=None):
        if pos_ids is None:
            return self.rope_cos[:T][None, None], self.rope_sin[:T][None, None]
        return self.rope_cos[pos_ids][:, None], self.rope_sin[pos_ids][:, None]   # (B, 1, T, hd/2)

    def hidden(self, idx):
        """idx (B, T) -> residual stream (B, T, D) that all MTP heads read"""
        cos, sin = self._rope(idx.size(1))
        x = self.drop(self.embed(idx))
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x

    def head_z(self, x, k):
        """normalised features of MTP head k (0-based; head 0 = next token)"""
        return self.heads[k](x)

    def forward(self, idx):
        """next-token logits (B, T, V) from head 1 - the path decoding uses; training goes through hidden() + mtp heads"""
        return F.linear(self.head_z(self.hidden(idx), 0), self.embed.weight)

    def out_logits(self, z):
        """(Phase 6) logits over the real_vocab tokens from normalised features z (the padding rows are left out)"""
        return F.linear(z, self.embed.weight[: self.cfg.real_vocab])

    def ban_tensor(self, device):
        """ids that must never be generated: the 5 specials plus the padding rows of the embedding"""
        if self._ban is None or self._ban.device != device:
            c = self.cfg
            ids = sorted({c.pad_id, c.unk_id, c.eos_id, c.bos_id, c.mask_id} | set(range(c.real_vocab, c.vocab_size)))
            self._ban = torch.tensor(ids, dtype=torch.long, device=device)
        return self._ban

    # ---- KV-cached generation (head 1 only) ----
    def init_cache(self, B, total_len, dtype, device):
        kv, hd = self.cfg.n_kv_heads, self.cfg.d_model // self.cfg.n_heads
        return [{"k": torch.zeros(B, kv, total_len, hd, dtype=dtype, device=device),
                 "v": torch.zeros(B, kv, total_len, hd, dtype=dtype, device=device)}
                for _ in self.blocks]

    def forward_cached(self, idx, pos_ids, cache, start, attn_mask):
        """idx (B, T) new tokens with explicit RoPE positions pos_ids (B, T); returns next-token logits of the last one"""
        cos, sin = self._rope(idx.size(1), pos_ids)
        x = self.embed(idx)
        for blk, c in zip(self.blocks, cache):
            x = blk.forward_cached(x, cos, sin, c, start, attn_mask)
        return F.linear(self.head_z(x[:, -1], 0), self.embed.weight)


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def load_teacher_base(device):
    """Phase-1 Model 3 as pretrained (the Phase-6 adapter is attached later, after the tokenizer check), frozen, eval mode"""
    ckpt = torch.load(find_ckpt(TEACHER_TAG, TEACHER_CKPT, "teacher"), map_location="cpu", weights_only=False)
    fields = M3Config.__dataclass_fields__
    cfg = M3Config(**{k: v for k, v in ckpt["config"].items() if k in fields})
    cfg.dropout = 0.0
    model = M3DecoderLM(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.pretrain_val_loss = float(ckpt.get("val_loss", ckpt.get("best_val", float("nan"))))
    log(f"Phase-1 Model 3 (teacher) loaded: {count_params(model) / 1e6:.1f}M params, d_model {cfg.d_model}, {cfg.n_layers} layers, "
        f"{cfg.mtp_heads} MTP heads, vocab {cfg.real_vocab} (+{cfg.vocab_size - cfg.real_vocab} padding rows), "
        f"next-token val loss {model.pretrain_val_loss:.4f}")
    del ckpt
    return freeze(model.to(device)), cfg




# ------------------------------- STUDENT: Model 4 (verbatim) -------------------------------
# architecture constants of the Model-4 notebook (only the DEFAULTS of ModelConfig; the real values come from the checkpoint)
VOCAB_SIZE = 50_000
D_MODEL = 512
HEAD_DIM = 64
N_HEADS = 8
N_KV_HEADS = 2                     # GQA 4:1 -> q dim 512, kv dim 128
FFN_HIDDEN = 1792
ATTN_DROPOUT = 0.0
FFN_DROPOUT = 0.10                 # dropout on the FFN output (before the residual add)
WINDOW_SMALL = 128                 # local attention, small window
WINDOW_LARGE = 256                 # local attention, large window
LAYER_TYPES = ["local_small", "local_large", "memory"] * 3      # L0..L8 = 3 x [Local-128 -> Local-256 -> Memory-Global]
N_LAYERS = len(LAYER_TYPES)        # 9
MEM_CHUNKS = 8                     # the 1024-token context is cut into 8 chunks ...
MEM_CHUNK_SIZE = 128               # ... of 128 tokens
MEM_PER_CHUNK = 4                  # 4 learned memory tokens summarise one chunk
N_MEM = MEM_CHUNKS * MEM_PER_CHUNK                # 32 memory tokens of dimension D_MODEL
LOCAL_IMPL = {"mode": "block"}     # "block" = block-wise sliding window (fast); the start-up self-test switches to "mask" if
                                   # it ever disagrees with the plain masked reference
MTP_DEPTH = 3                              # head k predicts token t+k, k = 1..3 (head 1 = the normal next-token head)
MTP_WEIGHTS = (1.0, 0.5, 0.25)
MTP_POS_FRAC = 0.25
assert D_MODEL == N_HEADS * HEAD_DIM and N_HEADS % N_KV_HEADS == 0, "heads / head dim do not tile d_model"
assert MEM_CHUNKS * MEM_CHUNK_SIZE == CONTEXT_LENGTH, "memory chunks must tile the context"


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    n_kv_heads: int = N_KV_HEADS
    ffn_hidden: int = FFN_HIDDEN
    attn_dropout: float = ATTN_DROPOUT
    ffn_dropout: float = FFN_DROPOUT
    layer_types: tuple = tuple(LAYER_TYPES)       # one entry per transformer block
    window_small: int = WINDOW_SMALL
    window_large: int = WINDOW_LARGE
    n_mem_chunks: int = MEM_CHUNKS
    chunk_size: int = MEM_CHUNK_SIZE
    mem_per_chunk: int = MEM_PER_CHUNK
    max_len: int = CONTEXT_LENGTH + 64
    pad_id: int = 0
    unk_id: int = 1
    eos_id: int = 2
    n_mtp: int = MTP_DEPTH
    mtp_weights: tuple = MTP_WEIGHTS
    mtp_pos_frac: float = MTP_POS_FRAC

    @property
    def n_layers(self):
        return len(self.layer_types)


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
    return freqs.cos(), freqs.sin()          # each (max_len, head_dim/2)


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); cos/sin: (1 or B, 1, T, hd/2)
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.type_as(x)


def repeat_kv(x, groups):
    # (B, KV, T, hd) -> (B, KV*groups, T, hd); query head h reads kv head h // groups
    if groups == 1:
        return x
    B, KV, T, hd = x.shape
    return x[:, :, None].expand(B, KV, groups, T, hd).reshape(B, KV * groups, T, hd)


_BLOCK_MASKS = {}


def block_local_mask(window, n_blocks, device):
    """(n_blocks, 1, W, 2W) bool. A query i of a block looks at the 2W keys [previous block | own block]; key j is allowed iff
    i < j <= i + W, i.e. the token itself and the W-1 tokens before it. The first block has no previous block."""
    key = (window, n_blocks, str(device))
    m = _BLOCK_MASKS.get(key)
    if m is None:
        i = torch.arange(window, device=device)[:, None]
        j = torch.arange(2 * window, device=device)[None, :]
        base = (j > i) & (j <= i + window)                   # (W, 2W)
        m = base[None].repeat(n_blocks, 1, 1)                # (nb, W, 2W)
        m[0] &= (j >= window)                                # no previous block in front of block 0
        m = m[:, None]
        _BLOCK_MASKS[key] = m
    return m


def local_attention(q, k, v, window, dropout_p=0.0):
    """Exact causal sliding-window attention (a token sees itself and the window-1 tokens before it), computed block-wise:
    the queries of block n only meet the keys of blocks n-1 and n (2*window keys) instead of all T keys.
    q, k, v: (B, H, T, hd), T arbitrary (padded up to a multiple of window internally)."""
    B, H, T, hd = q.shape
    nb = (T + window - 1) // window
    Tp = nb * window
    if Tp != T:                          # padded keys sit after every real query, so causality hides them
        q, k, v = (F.pad(t, (0, 0, 0, Tp - T)) for t in (q, k, v))
    qb = q.reshape(B, H, nb, window, hd).transpose(1, 2).reshape(B * nb, H, window, hd)

    def two_blocks(t):                   # (B, H, Tp, hd) -> (B*nb, H, 2W, hd) = [previous block | own block]
        tb = t.reshape(B, H, nb, window, hd)
        prev = torch.cat([tb.new_zeros(B, H, 1, window, hd), tb[:, :, :-1]], dim=2)
        return torch.cat([prev, tb], dim=3).transpose(1, 2).reshape(B * nb, H, 2 * window, hd)

    mask = block_local_mask(window, nb, q.device)                                    # (nb, 1, W, 2W)
    mask = mask[None].expand(B, nb, 1, window, 2 * window).reshape(B * nb, 1, window, 2 * window)
    out = F.scaled_dot_product_attention(qb, two_blocks(k), two_blocks(v), attn_mask=mask, dropout_p=dropout_p)
    return out.reshape(B, nb, H, window, hd).transpose(1, 2).reshape(B, H, Tp, hd)[:, :, :T]


def local_attention_ref(q, k, v, window, dropout_p=0.0):
    """plain masked SDPA over all T keys - the slow reference (and the fallback) of local_attention"""
    T = q.size(2)
    i = torch.arange(T, device=q.device)
    mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)


class GQAttention(nn.Module):
    """causal grouped-query self-attention with RoPE, restricted to a sliding window (10 query heads share 2 key/value heads)"""

    def __init__(self, cfg, window):
        super().__init__()
        self.h = cfg.n_heads
        self.kv = cfg.n_kv_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.groups = self.h // self.kv
        self.window = window
        self.p = cfg.attn_dropout
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
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k, v = repeat_kv(k, self.groups), repeat_kv(v, self.groups)
        p = self.p if self.training else 0.0
        attend = local_attention if LOCAL_IMPL["mode"] == "block" else local_attention_ref
        out = attend(q, k, v, self.window, p)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))

    def forward_cached(self, x, cos, sin, k_cache, v_cache, start, attn_mask, pos_ids, kpos):
        """T new tokens go into cache slots [start, start+T); they attend to slots [0, start+T). attn_mask carries causality +
        validity in slot space; the window is applied on the REAL positions (pos_ids of the queries, kpos of every cached slot),
        because left-padded rows have different slot -> position offsets."""
        B, T, _ = x.shape
        q, k, v = self._qkv(x)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k_cache[:, :, start:start + T] = k.to(k_cache.dtype)
        v_cache[:, :, start:start + T] = v.to(v_cache.dtype)
        keys = repeat_kv(k_cache[:, :, :start + T], self.groups).to(q.dtype)
        vals = repeat_kv(v_cache[:, :, :start + T], self.groups).to(q.dtype)
        in_window = kpos[:, None, None, :] > (pos_ids[:, None, :, None] - self.window)       # (B, 1, T, S)
        out = F.scaled_dot_product_attention(q, keys, vals, attn_mask=attn_mask & in_window)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LocalBlock(nn.Module):
    """pre-norm block: sliding-window GQA attention (RoPE) + SwiGLU"""
    kind = "local"

    def __init__(self, cfg, window):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = GQAttention(cfg, window)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.ffn_dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

    def forward_cached(self, x, cos, sin, cache, start, attn_mask, pos_ids, kpos):
        x = x + self.attn.forward_cached(self.norm1(x), cos, sin, cache["k"], cache["v"], start, attn_mask, pos_ids, kpos)
        x = x + self.ffn(self.norm2(x))
        return x


def memory_masks(q_chunk, k_chunk, k_valid, mem_per_chunk, n_slots):
    """Boolean masks of the memory layers, all slots / all rows at once.
        q_chunk (Bq, Tq)        chunk id of every query token
        k_chunk (Bk, S), k_valid (Bk, S)   chunk id / real-token flag of every key token (cached slots)
    returns
        w   (B, 1, slots, S)    WRITE: the memory tokens of chunk c may read exactly the (real) tokens of chunk c
        r   (B, 1, Tq, slots)   READ : a token of chunk j may read the memory tokens of the chunks < j  (strictly earlier = causal)
        has (B, Tq)             False for tokens with nothing to read (first chunk, padding rows): their memory output is zeroed
    Rows that would be empty (slots of chunks that have no token yet, tokens of chunk 0) get a harmless all-allowed fallback so
    that no softmax row is empty (NaN); the fallback slots are never read by a token, and the zeroed rows contribute nothing."""
    slot_chunk = torch.arange(n_slots, device=q_chunk.device) // mem_per_chunk                 # (slots,)
    w = (k_chunk[:, None, :] == slot_chunk[None, :, None]) & k_valid[:, None, :]               # (Bk, slots, S)
    w = torch.where(w.any(-1, keepdim=True), w, k_valid[:, None, :])
    r = slot_chunk[None, None, :] < q_chunk[:, :, None]                                        # (Bq, Tq, slots)
    has = r.any(-1)
    r = torch.where(has[..., None], r, torch.ones_like(r))
    return w[:, None], r[:, None], has


class MemoryBlock(nn.Module):
    """Memory-Global layer. WRITE (memory tokens <- the tokens of their own chunk, gated update of the memory carried over from
    the previous stage), READ (tokens <- memory tokens of strictly earlier chunks, GQA, no RoPE), then SwiGLU. Returns (x, memory)."""
    kind = "memory"

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.h = cfg.n_heads
        self.kv = cfg.n_kv_heads
        self.hd = d // self.h
        self.groups = self.h // self.kv
        self.p = cfg.attn_dropout
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.mem_norm_w = RMSNorm(d)                 # norm of the memory as query of the write attention
        self.mem_norm_r = RMSNorm(d)                 # norm of the memory as key/value of the read attention
        # write: memory tokens (queries) <- chunk tokens (keys / values), no RoPE
        self.wq_proj = nn.Linear(d, self.h * self.hd, bias=False)
        self.wk_proj = nn.Linear(d, self.kv * self.hd, bias=False)
        self.wv_proj = nn.Linear(d, self.kv * self.hd, bias=False)
        self.write_out = nn.Linear(self.h * self.hd, d, bias=False)
        self.gate_proj = nn.Linear(2 * d, d, bias=False)            # g = sigmoid(W [M_prev ; candidate])
        # read: tokens (queries) <- memory tokens (keys / values), no RoPE
        self.q_proj = nn.Linear(d, self.h * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.kv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.h * self.hd, d, bias=False)
        self.ffn = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.ffn_dropout)

    def write_kv(self, xn):
        B, T, _ = xn.shape
        k = self.wk_proj(xn).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.wv_proj(xn).view(B, T, self.kv, self.hd).transpose(1, 2)
        return k, v

    def update(self, mem_prev, k_all, v_all, w_mask):
        """gated memory update. mem_prev (B, slots, D); k_all / v_all (B, KV, S, hd) of every token seen so far"""
        B, n, _ = mem_prev.shape
        q = self.wq_proj(self.mem_norm_w(mem_prev)).view(B, n, self.h, self.hd).transpose(1, 2)
        keys = repeat_kv(k_all, self.groups).to(q.dtype)
        vals = repeat_kv(v_all, self.groups).to(q.dtype)
        out = F.scaled_dot_product_attention(q, keys, vals, attn_mask=w_mask, dropout_p=self.p if self.training else 0.0)
        cand = self.write_out(out.transpose(1, 2).reshape(B, n, -1)).type_as(mem_prev)
        gate = torch.sigmoid(self.gate_proj(torch.cat([mem_prev, cand], dim=-1)))
        return mem_prev + gate * (cand - mem_prev)                    # = (1 - g) * M_prev + g * candidate

    def read(self, xn, mem, r_mask, has):
        B, T, _ = xn.shape
        n = mem.size(1)
        mn = self.mem_norm_r(mem)
        q = self.q_proj(xn).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(mn).view(B, n, self.kv, self.hd).transpose(1, 2)
        v = self.v_proj(mn).view(B, n, self.kv, self.hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, repeat_kv(k, self.groups), repeat_kv(v, self.groups), attn_mask=r_mask,
                                             dropout_p=self.p if self.training else 0.0)
        out = self.o_proj(out.transpose(1, 2).reshape(B, T, -1))
        return out * has[..., None].to(out.dtype)

    def forward(self, x, mem_prev, mctx):
        xn = self.norm1(x)
        k, v = self.write_kv(xn)
        mem = self.update(mem_prev, k, v, mctx["w"])
        x = x + self.read(xn, mem, mctx["r"], mctx["has"])
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, mem

    def forward_cached(self, x, mem_prev, cache, start, mctx):
        """the write keys/values of every token live in the cache (no RoPE), the memory is recomputed from them on every call;
        only completed chunks are readable, so this equals the full forward pass exactly"""
        xn = self.norm1(x)
        k, v = self.write_kv(xn)
        T = x.size(1)
        S = start + T
        cache["k"][:, :, start:S] = k.to(cache["k"].dtype)
        cache["v"][:, :, start:S] = v.to(cache["v"].dtype)
        mem = self.update(mem_prev, cache["k"][:, :, :S], cache["v"][:, :, :S], mctx["w"])
        x = x + self.read(xn, mem, mctx["r"], mctx["has"])
        x = x + self.ffn(self.norm2(x))
        return x, mem


class DecoderLM(nn.Module):
    """MSM transformer: 3 x [Local-128 -> Local-256 -> Memory-Global] with RoPE + GQA + SwiGLU and multi-token prediction.
    All heads read the same final hidden state h and share the tied embedding E as output matrix: head k (k = 1..K) predicts
    token t+k. Head 1 is the plain LM head (RMSNorm(h) @ E^T) and the only one used for validation and decoding; heads 2..K are
    training-time auxiliaries  RMSNorm_k(h + W_k h) @ E^T.
    Memory: mem_init (32 x D) is the learned latent memory. The first memory layer updates it from the tokens, the result is
    handed to the next memory layer (M0 -> M1 -> M2). Memory slot c always belongs to chunk c = position // 128."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_slots = cfg.n_mem_chunks * cfg.mem_per_chunk
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)       # also the (tied) output projection
        windows = {"local_small": cfg.window_small, "local_large": cfg.window_large}
        self.blocks = nn.ModuleList([MemoryBlock(cfg) if kind == "memory" else LocalBlock(cfg, windows[kind])
                                     for kind in cfg.layer_types])
        self.final_norm = RMSNorm(cfg.d_model)
        self.mem_init = nn.Parameter(torch.empty(self.n_slots, cfg.d_model))      # learned latent memory
        self.mtp_proj = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_model, bias=False) for _ in range(cfg.n_mtp - 1)])
        self.mtp_norm = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.n_mtp - 1)])
        cos, sin = build_rope(cfg.d_model // cfg.n_heads, cfg.max_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_parameter("eos_delta", None)     # (Phase 6) optional trainable correction of the EOS output row

        self.apply(self._init_weights)
        nn.init.normal_(self.mem_init, mean=0.0, std=0.02)
        # damp the residual-branch output projections (GPT-2 style)
        scale = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope(self, T, pos_ids=None):
        if pos_ids is None:
            return self.rope_cos[:T][None, None], self.rope_sin[:T][None, None]
        return self.rope_cos[pos_ids][:, None], self.rope_sin[pos_ids][:, None]   # (B, 1, T, hd/2)

    def _chunk_of(self, pos):
        c = self.cfg
        return (pos // c.chunk_size).clamp(max=c.n_mem_chunks - 1)

    def _mem_ctx_train(self, T, device):
        pos = torch.arange(T, device=device)[None]                  # (1, T): no padding in training
        chunk = self._chunk_of(pos)
        w, r, has = memory_masks(chunk, chunk, torch.ones_like(pos, dtype=torch.bool), self.cfg.mem_per_chunk, self.n_slots)
        return {"w": w, "r": r, "has": has}

    def trunk(self, idx):
        """(B, T) ids -> (B, T, D) final hidden states"""
        B, T = idx.shape
        cos, sin = self._rope(T)
        x = self.embed(idx)
        mem, mctx = None, None
        for blk in self.blocks:
            if blk.kind == "memory":
                if mem is None:
                    mem = self.mem_init[None].expand(B, -1, -1)       # the learned latent memory enters the first memory layer
                    mctx = self._mem_ctx_train(T, idx.device)
                x, mem = blk(x, mem, mctx)
            else:
                x = blk(x, cos, sin)
        return self.final_norm(x)

    def out_logits(self, z):
        """(Phase 6) logits of normalised features z through the tied embedding, plus the optional EOS-row correction"""
        out = F.linear(z, self.embed.weight)
        if self.eos_delta is not None:
            out[..., self.cfg.eos_id] += z @ self.eos_delta.to(z.dtype)
        return out

    def head_input(self, h, k):
        """hidden state as seen by head k (0-based: head 0 predicts the next token)"""
        if k == 0:
            return h
        return self.mtp_norm[k - 1](h + self.mtp_proj[k - 1](h))

    def forward(self, idx):
        """training / validation path of the next-token head: idx (B, T) -> logits (B, T, V)"""
        return self.out_logits(self.head_input(self.trunk(idx), 0))                  # weight tying: logits = h @ E^T

    # ---- KV-cached generation (next-token head only) ----
    def init_cache(self, B, total_len, dtype, device):
        """one {"k", "v"} per block: the RoPE'd keys/values of a local layer, the (un-rotated) WRITE keys/values of a memory layer.
        cache[0] also carries "pos" (B, total_len): the real position of every slot, -1 = padding / not filled yet."""
        kv, hd = self.cfg.n_kv_heads, self.cfg.d_model // self.cfg.n_heads
        cache = [{"k": torch.zeros(B, kv, total_len, hd, dtype=dtype, device=device),
                  "v": torch.zeros(B, kv, total_len, hd, dtype=dtype, device=device)}
                 for _ in self.blocks]
        cache[0]["pos"] = torch.full((B, total_len), -1, dtype=torch.long, device=device)
        return cache

    def forward_cached(self, idx, pos_ids, cache, start, attn_mask):
        """idx (B, T) new tokens with explicit real positions pos_ids (B, T); returns logits of the last one.
        attn_mask (B, 1, T, start+T) is built by start_generation / next_logits; its LAST query row says which of the
        slots [0, start+T) hold real tokens (the last query is always a real token), which is how padding slots are recognised."""
        B, T = idx.shape
        S = start + T
        c = self.cfg
        cos, sin = self._rope(T, pos_ids)
        pos_store = cache[0]["pos"]
        new_real = attn_mask[:, 0, -1, start:S]
        pos_store[:, start:S] = torch.where(new_real, pos_ids, torch.full_like(pos_ids, -1))
        kpos = pos_store[:, :S]                                      # (B, S) real position of every cached slot, -1 = padding
        x = self.embed(idx)
        mem, mctx = None, None
        for blk, ca in zip(self.blocks, cache):
            if blk.kind == "memory":
                if mem is None:
                    mem = self.mem_init[None].expand(B, -1, -1)
                    w, r, has = memory_masks(self._chunk_of(pos_ids), self._chunk_of(kpos.clamp(min=0)), kpos >= 0,
                                             c.mem_per_chunk, self.n_slots)
                    mctx = {"w": w, "r": r, "has": has}
                x, mem = blk.forward_cached(x, mem, ca, start, mctx)
            else:
                x = blk.forward_cached(x, cos, sin, ca, start, attn_mask, pos_ids, kpos)
        x = self.final_norm(x)
        return self.out_logits(self.head_input(x[:, -1], 0))


def load_student(device):
    """Phase-1 Model 4 (the base of the LoRA student); FFN dropout set to FT_DROPOUT"""
    ckpt = torch.load(find_ckpt(STUDENT_TAG, STUDENT_CKPT, "student"), map_location="cpu", weights_only=False)
    fields = ModelConfig.__dataclass_fields__
    c = {k: v for k, v in ckpt["config"].items() if k in fields}
    for k in ("layer_types", "mtp_weights"):
        if k in c:
            c[k] = tuple(c[k])
    cfg = ModelConfig(**c)
    cfg.ffn_dropout = FT_DROPOUT
    model = DecoderLM(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.pretrain_val_loss = float(ckpt.get("val_loss", ckpt.get("best_val", float("nan"))))
    log(f"Phase-1 Model 4 (student) loaded: {count_params(model) / 1e6:.1f}M params, d_model {cfg.d_model}, {cfg.n_layers} layers "
        f"({'/'.join(k.replace('local_small', 'L128').replace('local_large', 'L256').replace('memory', 'MEM') for k in cfg.layer_types)}), "
        f"{model.n_slots} memory tokens, vocab {cfg.vocab_size}, next-token val loss {model.pretrain_val_loss:.4f}")
    del ckpt
    return model.to(device), cfg


# =====================================================================================
# STUDENT ADAPTER: LoRA on Model 4 (own implementation, no extra dependency)
#   W' = W0 + (alpha / r) B A   (A: r x in, B: out x r, B = 0 at init -> the student starts as the exact Phase-1 Model 4)
#   + a trainable 512-dim correction of the EOS output row (TRAIN_EOS_ROW), the learned latent memory and the final-norm gain
#   (the latter two directly, at BASE_LR_MULT x LR).  After training everything is folded into the Phase-1 weights.
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


def lora_targets():
    return tuple(LORA_TARGETS) + (tuple(LORA_MEM_TARGETS) if ADAPT_MEMORY else ())


def lora_settings():
    return {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": list(lora_targets()),
            "train_norms": TRAIN_NORMS, "train_eos": TRAIN_EOS_ROW, "train_mem_init": TRAIN_MEM_INIT,
            "train_final_norm": TRAIN_FINAL_NORM}


def kd_settings():
    """what must not change between the sessions of one run (the teacher itself is only logged: see train())"""
    return {"use_teacher": USE_TEACHER, "alpha": KD_ALPHA if USE_TEACHER else 0.0, "temp": KD_TEMP if USE_TEACHER else 1.0}


def apply_lora(model, s):
    dev = next(model.parameters()).device
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
    if s.get("train_final_norm"):
        model.final_norm.weight.requires_grad = True
    if s.get("train_mem_init"):
        model.mem_init.requires_grad = True
    if s.get("train_eos"):
        model.eos_delta = nn.Parameter(torch.zeros(model.cfg.d_model))
    model.to(dev)
    total, trainable = count_params(model), count_params(model, trainable_only=True)
    extras = [x for x, on in (("learned latent memory", s.get("train_mem_init")), ("final-norm gain", s.get("train_final_norm")),
                              ("%d-dim EOS output row" % model.cfg.d_model, s.get("train_eos"))) if on]
    log(f"LoRA applied to {n} linear layers (r={s['r']}, alpha={s['alpha']}): "
        f"{trainable / 1e6:.2f}M trainable / {total / 1e6:.1f}M total ({100 * trainable / total:.2f}%)"
        f"{' | + trainable: ' + ', '.join(extras) if extras else ''}")


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
    """fold W + scaling * B @ A into the base weights, the EOS correction into the tied embedding row"""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                w = child.base.weight
                w += (child.scaling * (child.lora_B.float() @ child.lora_A.float())).to(w.dtype)
                setattr(module, name, child.base)
    if getattr(model, "eos_delta", None) is not None:
        model.embed.weight[model.cfg.eos_id] += model.eos_delta.to(model.embed.weight.dtype)
        model.eos_delta = None
    for p in model.parameters():
        p.requires_grad = False
    log("LoRA merged into the base weights")


@torch.no_grad()
def lora_report(model):
    """how large is the learned update  dW = (alpha/r) B A  relative to W0, per module type (material for the report)"""
    rel = {}
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear):
            dw = m.scaling * (m.lora_B.float() @ m.lora_A.float())
            rel.setdefault(n.split(".")[-1], []).append((dw.norm() / m.base.weight.float().norm().clamp_min(1e-12)).item())
    if not rel:
        return None
    out = {"rel_update_norm": {k: float(np.mean(v)) for k, v in rel.items()}}
    if getattr(model, "eos_delta", None) is not None:
        out["eos_delta_norm"] = float(model.eos_delta.float().norm())
    log("LoRA update ||dW||_F/||W0||_F per module type: " + ", ".join(f"{k} {v:.4f}" for k, v in out["rel_update_norm"].items()))
    return out


# =====================================================================================
# TEACHER ADAPTER: LoRMA on Model 3  (arXiv 2506.07621; the SAME classes as phase6_lorma_model3.py, which trained the adapter)
#   W' = I(BA) . W0, A (r x d), B (d x r) act on the OUTPUT of a frozen layer (d = its output size).
#     "plus": I(BA) = (alpha/r) BA + I               y' = y + (alpha/r) B (A y)
#     "pi"  : I(BA) = row i of BA rotated by i        y' = y + (I_pi(BA) - I) y
#   Only needed to REBUILD the fine-tuned teacher: the adapter is loaded and folded into the Phase-1 weights before distillation.
# =====================================================================================
def rotation_index(d):
    """idx[i, j] = (j - i) mod d, so that  M.gather(1, idx)[i, j] = M[i, (j - i) mod d]  (= row i of M shifted right by i)"""
    ar = torch.arange(d)
    return (ar[None, :] - ar[:, None]) % d


class LoRMALinear(nn.Module):
    def __init__(self, base, r, alpha, variant, dropout):
        super().__init__()
        assert variant in ("plus", "pi"), f"unknown LoRMA variant {variant!r}"
        self.base = base
        d = base.out_features
        self.r, self.variant, self.scaling = r, variant, alpha / r
        self.lorma_A = nn.Parameter(torch.zeros(r, d))
        self.lorma_B = nn.Parameter(torch.zeros(d, r))
        if variant == "plus":
            nn.init.kaiming_uniform_(self.lorma_A, a=math.sqrt(5))        # B = 0  =>  I(BA) = I
        else:
            nn.init.kaiming_uniform_(self.lorma_B, a=math.sqrt(5))
            with torch.no_grad():
                self.lorma_B[:, 0] = 1.0                                   # A is all-zero except A[0, 0] = 1, so that
                self.lorma_A[0, 0] = 1.0                                   # BA = "ones in column 0" -> rotated by i = identity
            self.register_buffer("rot_index", rotation_index(d), persistent=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def delta_operator(self):
        """(d x d) fp32 matrix  I(BA) - I :  adapted output = y + delta @ y ,  merged weight = W0 + delta @ W0"""
        with torch.autocast(device_type="cuda", enabled=False):
            m = self.lorma_B.float() @ self.lorma_A.float()
            if self.variant == "plus":
                return m * self.scaling
            return m.gather(1, self.rot_index) - torch.eye(m.size(0), device=m.device, dtype=m.dtype)

    def forward(self, x):
        y = self.base(x)
        if self.variant == "plus":                                          # factored form: two skinny matmuls
            return y + F.linear(F.linear(self.drop(y), self.lorma_A), self.lorma_B) * self.scaling
        return y + F.linear(self.drop(y), self.delta_operator())            # pi: the rotated operator is formed explicitly


def apply_lorma(model, s):
    dev = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad = False
    n = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in s["targets"]:
                setattr(module, name, LoRMALinear(child, s["r"], s["alpha"], s["variant"], s["dropout"]))
                n += 1
    if s.get("train_norms"):
        for m in model.modules():
            if isinstance(m, M3RMSNorm):
                m.weight.requires_grad = True
    if s.get("train_eos"):
        model.eos_delta = nn.Parameter(torch.zeros(model.cfg.d_model))
    model.to(dev)
    return n


@torch.no_grad()
def merge_lorma(model):
    """W_ft = I(BA) . W0 = W0 + delta @ W0 (computed in fp32), EOS correction folded into the tied embedding row"""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRMALinear):
                w = child.base.weight
                w += (child.delta_operator() @ w.float()).to(w.dtype)
                setattr(module, name, child.base)
    if getattr(model, "eos_delta", None) is not None:
        model.embed.weight[model.cfg.eos_id] += model.eos_delta.to(model.embed.weight.dtype)
        model.eos_delta = None
    for p in model.parameters():
        p.requires_grad = False


def find_teacher_adapter():
    """(val_loss, path, run_tag) of the best Phase-6 Model-3 LoRMA adapter (lowest validation loss), or None"""
    if TEACHER_ADAPTER:
        if not os.path.exists(TEACHER_ADAPTER):
            raise FileNotFoundError(f"TEACHER_ADAPTER = {TEACHER_ADAPTER} does not exist")
        meta = read_json(os.path.join(os.path.dirname(TEACHER_ADAPTER), "lorma_best_meta.json")) or {}
        return float(meta.get("val_loss", float("nan"))), TEACHER_ADAPTER, meta.get("tag", "?")
    best = None
    for d in walk_find("lorma_best_meta.json", [INPUT_ROOT, os.getcwd()]):
        meta = read_json(os.path.join(d, "lorma_best_meta.json")) or {}
        tag = str(meta.get("tag", ""))
        if not tag.startswith(TEACHER_RUN_PREFIX) or tag.endswith("_smoke") or not os.path.exists(os.path.join(d, "lorma_best.pt")):
            continue
        vl = float(meta.get("val_loss", float("nan")))
        key = vl if math.isfinite(vl) else 1e9
        if best is None or key < best[0]:
            best = (key, os.path.join(d, "lorma_best.pt"), tag, vl)
    return None if best is None else (best[3], best[1], best[2])


def attach_teacher_adapter(model):
    """fold the Phase-6 LoRMA adapter into the Phase-1 Model 3 -> the fine-tuned teacher. Returns a description of the teacher."""
    found = find_teacher_adapter()
    if found is None:
        msg = (f"no fine-tuned Model-3 teacher found: no lorma_best.pt + lorma_best_meta.json with a RUN_TAG starting with "
               f"'{TEACHER_RUN_PREFIX}' under {INPUT_ROOT}. Attach the output of the Phase-6 Model-3 LoRMA run "
               f"(phase6_lorma_model3.py), or set TEACHER_ADAPTER to its lorma_best.pt.")
        if not ALLOW_PRETRAINED_TEACHER:
            raise FileNotFoundError(msg + " (A Phase-1 Model 3 that was never fine-tuned only continues the document, it does not "
                                          "summarise - set ALLOW_PRETRAINED_TEACHER = True to distil from it anyway.)")
        log("WARNING: " + msg + "\n  ALLOW_PRETRAINED_TEACHER is on -> the teacher is the Phase-1 Model 3 as pretrained: its soft "
                                "targets are those of a plain language model (EOS positions: hard labels only).")
        return {"kind": "pretrained", "eos_ok": False, "run_tag": None, "adapter_val_loss": float("nan"), "path": None}
    vl, path, tag = found
    ck = torch.load(path, map_location="cpu", weights_only=False)
    s = ck["lorma_cfg"]
    n = apply_lorma(model, s)
    load_adapter_state(model, ck["lorma"])
    merge_lorma(model)
    freeze(model)
    vl = float(ck.get("val_loss", vl)) if math.isfinite(float(ck.get("val_loss", float("nan")))) else vl
    info = {"kind": "finetuned", "eos_ok": bool(s.get("train_eos")), "run_tag": tag, "variant": s.get("variant"),
            "adapter_val_loss": vl, "path": path, "epochs_done": ck.get("epochs_done")}
    log(f"teacher = Phase-1 Model 3 + Phase-6 LoRMA-{s.get('variant')} adapter '{tag}' ({n} layers merged, adapter validation CE "
        f"{vl:.4f}, EOS row {'trained' if info['eos_ok'] else 'NOT trained'}) from {path}")
    del ck
    return info


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
    return tuple(STATE["tok"].encode(f"[part {k + 1}/{n}]"))


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
           "tok": getattr(tok, "fingerprint", ""),
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


def kd_loss_parts(s, t, y, temp, eos_id, eos_ok):
    """s, t: (N, V) fp32 student / teacher logits of the N summary positions, y: (N,) reference tokens.
    returns (ce_sum, kl_sum, agree):
        ce_sum  = sum of the cross-entropies against the reference tokens (all N positions)
        kl_sum  = T^2 * sum over positions of KL( softmax(t/T) || softmax(s/T) )   (the distillation term, summed)
        agree   = number of positions where student and teacher have the same arg-max token
    eos_ok False (teacher that never learned EOS): the EOS column is left out of both distributions and the positions whose
    reference token is EOS are skipped in the KL term, so the teacher can not push the student's EOS probability down."""
    assert s.shape == t.shape, f"student logits {tuple(s.shape)} vs teacher logits {tuple(t.shape)} - not the same vocabulary"
    ce = F.cross_entropy(s, y, reduction="sum")
    with torch.no_grad():
        agree = int((s.argmax(-1) == t.argmax(-1)).sum())
    if not eos_ok:
        col = torch.zeros(s.size(-1), dtype=torch.bool, device=s.device)
        col[eos_id] = True
        keep = y != eos_id
        s = s[keep].masked_fill(col, -1e4)
        t = t[keep].masked_fill(col, -1e4)
    ls = F.log_softmax(s / temp, dim=-1)
    lt = F.log_softmax(t / temp, dim=-1)
    kl = (lt.exp() * (lt - ls)).sum(-1)                 # >= 0, one value per position; the teacher side carries no gradient
    return ce, kl.sum() * (temp * temp), agree


def summary_loss_sum(model, x, y, teacher=None):
    """summed loss over the summary positions (labels != -100) of a right-padded batch.
    teacher None : plain cross-entropy
    teacher given: (1 - KD_ALPHA) * CE + KD_ALPHA * T^2 * KL(teacher || student)
    returns (loss_sum, n_positions, info) with info = {"ce": ..., "kl": ..., "agree": ...} (floats of the unweighted parts)"""
    sel = y != -100
    ys = y[sel]
    n = int(sel.sum().item())
    with amp_ctx():
        z = lm_features(model, x)
        s = lm_logits(model, z[sel])
    s = s.float()
    if teacher is None:
        ce = F.cross_entropy(s, ys, reduction="sum")
        return ce, n, {"ce": ce.item(), "kl": 0.0, "agree": 0}
    with torch.no_grad(), amp_ctx():
        t = lm_logits(teacher, lm_features(teacher, x)[sel])
    ce, kl, agree = kd_loss_parts(s, t.float(), ys, KD_TEMP, model.cfg.eos_id, STATE.get("teacher_eos_ok", False))
    loss = (1.0 - KD_ALPHA) * ce + KD_ALPHA * kl
    return loss, n, {"ce": ce.item(), "kl": kl.item(), "agree": agree}


@torch.no_grad()
def eval_loss(model, samples, device, teacher=None, full=False):
    """mean cross-entropy (hard labels) of the model on the summary positions of `samples`.
    full=True (needs a teacher): dict with the CE, the mean distillation term and the top-1 agreement with the teacher"""
    model.eval()
    order = np.argsort([-(len(p) + len(t)) for p, t in samples])
    ce_tot = kl_tot = 0.0
    cnt = agree = 0
    for i in range(0, len(order), EVAL_BS):
        x, y = collate([samples[j] for j in order[i:i + EVAL_BS]], model.cfg.pad_id)
        _, n, info = summary_loss_sum(model, x.to(device), y.to(device), teacher if full else None)
        if full:
            ce_tot += info["ce"]
            kl_tot += info["kl"]
            agree += info["agree"]
        else:
            ce_tot += info["ce"]
        cnt += n
    ce = ce_tot / max(cnt, 1)
    if not full:
        return ce
    return {"ce": ce, "kl": kl_tot / max(cnt, 1), "agree": agree / max(cnt, 1)}


def make_optimizer(model):
    """LoRA matrices (weight decay) | EOS row | the directly trained base parameters (latent memory, norm gains) at BASE_LR_MULT x LR"""
    lora, eos, base = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (eos if n == "eos_delta" else lora if "lora_" in n else base).append(p)
    groups = [{"params": lora, "weight_decay": WEIGHT_DECAY, "mult": 1.0},
              {"params": eos, "weight_decay": 0.0, "mult": 1.0},
              {"params": base, "weight_decay": 0.0, "mult": BASE_LR_MULT}]
    groups = [dict(g, lr=LR * g["mult"], base_lr=LR * g["mult"]) for g in groups if g["params"]]
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
    for d in walk_find("student_latest_meta.json", roots or [CKPT_DIR, INPUT_ROOT]):
        meta = read_json(os.path.join(d, "student_latest_meta.json"))
        if not meta or meta.get("tag") != RUN_TAG or not os.path.exists(os.path.join(d, "student_latest.pt")):
            continue
        if best is None or meta.get("global_step", -1) > best[1].get("global_step", -1):
            best = (d, meta)
    return best


def carry_over_best(roots=None):
    """copy the best adapter of an earlier session (attached as an input) into this session's folder if it is better"""
    cur = read_json(f"{CKPT_DIR}/student_best_meta.json")
    for d in walk_find("student_best_meta.json", roots or [INPUT_ROOT]):
        meta = read_json(os.path.join(d, "student_best_meta.json"))
        if not meta or meta.get("tag") != RUN_TAG or not os.path.exists(os.path.join(d, "student_best.pt")):
            continue
        if cur is None or meta["val_loss"] < cur["val_loss"]:
            os.makedirs(CKPT_DIR, exist_ok=True)
            shutil.copy(os.path.join(d, "student_best.pt"), f"{CKPT_DIR}/student_best.pt.tmp")
            os.replace(f"{CKPT_DIR}/student_best.pt.tmp", f"{CKPT_DIR}/student_best.pt")
            shutil.copy(os.path.join(d, "student_best_meta.json"), f"{CKPT_DIR}/student_best_meta.json")
            cur = meta
    return cur


# ---------------------------- training ----------------------------
def train(model, teacher, train_s, val_s, device, t_start, stop_at_step=None):
    """LoRA (+ distillation from `teacher`, None = plain CE) training of the student with mid-epoch checkpoints. Returns a dict: finished (all EPOCHS done), resumed_from
    ((epoch, batch, global_step) of the checkpoint or None), final_step, best_val.
    stop_at_step: simulate a time-out after this many optimizer steps (used by the smoke test)."""
    best_path, best_meta_path = f"{CKPT_DIR}/student_best.pt", f"{CKPT_DIR}/student_best_meta.json"
    latest_path, latest_meta_path = f"{CKPT_DIR}/student_latest.pt", f"{CKPT_DIR}/student_latest_meta.json"
    best_meta = carry_over_best()
    best_val = best_meta["val_loss"] if best_meta else float("inf")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(model)
    base_lrs = [g["base_lr"] for g in opt.param_groups]
    scaler = make_grad_scaler(AMP["enabled"] and AMP["dtype"] == torch.float16)
    n = len(train_s)
    steps_per_epoch = max(1, n // EFF_BS)
    total_steps = EPOCHS * steps_per_epoch
    state = {"epoch": 0, "bi": 0, "global_step": 0, "train_sec": 0.0, "ep_loss": 0.0, "ep_n": 0, "ep_ce": 0.0}
    val_extra = {}
    history, baseline, ck_ema, ck_sched, val_check, resumed_from = [], None, None, None, None, None

    prev = find_previous_latest()
    if prev is not None:
        d, _ = prev
        ck = torch.load(os.path.join(d, "student_latest.pt"), map_location="cpu", weights_only=False)
        if ck.get("lora_cfg") != lora_settings():
            raise ValueError(f"the checkpoint in {d} was trained with LoRA settings {ck.get('lora_cfg')} but the current ones are "
                             f"{lora_settings()} - change RUN_TAG for a new run, or restore the old settings")
        if ck.get("kd_cfg") != kd_settings():
            raise ValueError(f"the checkpoint in {d} was trained with distillation settings {ck.get('kd_cfg')} but the current ones are "
                             f"{kd_settings()} - change KD_ALPHA / KD_TEMP back, or use a new RUN_TAG for a new run")
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
            state.update(bi=0, ep_loss=0.0, ep_n=0, ep_ce=0.0)
        ck_ema, ck_sched, val_check = ck.get("ema"), ck.get("sched"), ck.get("val_check")
        history, baseline = ck.get("history", []), ck.get("baseline")
        resumed_from = (state["epoch"], state["bi"], state["global_step"])
        log(f"RESUMED from {d}: epoch {state['epoch'] + 1}, batch {state['bi']}, step {state['global_step']}, "
            f"train time so far {fmt_time(state['train_sec'])}")
        del ck
    elif REQUIRE_RESUME:
        seen = [(d, (read_json(os.path.join(d, "student_latest_meta.json")) or {}).get("tag"))
                for d in walk_find("student_latest_meta.json", [INPUT_ROOT])]
        raise FileNotFoundError(
            f"REQUIRE_RESUME is on, but no earlier checkpoint with tag '{RUN_TAG}' was found under {INPUT_ROOT}. "
            f"student_latest_meta.json files seen: {seen or 'none'}. Add the earlier run's output (checkpoints/student_latest.pt + "
            "student_latest_meta.json) as an input of this notebook and keep RUN_TAG unchanged - or set REQUIRE_RESUME = False.")

    ema = EMA(model, EMA_DECAY) if USE_EMA else None
    if ema is not None and ck_ema is not None:
        ema.load_state_dict(ck_ema)

    def ema_ctx():
        return ema.applied() if ema is not None else contextlib.nullcontext()

    if prev is None:                                           # reference point for the report: Phase-1 model, LoRA still a no-op
        baseline = eval_loss(model, val_s, device)
        history.append({"epoch": 0, "val_loss": baseline, "val_ppl": math.exp(min(baseline, 20))})
        log(f"validation CE of the Phase-1 Model 4 before adaptation: {baseline:.4f} (ppl {math.exp(min(baseline, 20)):.2f})")
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
        atomic_torch_save({"lora": trainable_state(model), "lora_cfg": lora_settings(), "kd_cfg": kd_settings(), "opt": opt.state_dict(),
                           "scaler": scaler.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                           **state, "sched": sched, "steps_per_epoch": steps_per_epoch, "history": history,
                           "baseline": baseline, "val_check": vcheck, "tag": RUN_TAG}, latest_path)
        atomic_json({"tag": RUN_TAG, "epoch": state["epoch"], "bi": state["bi"], "global_step": state["global_step"],
                     "train_sec": state["train_sec"], "steps_per_epoch": steps_per_epoch, "sched": sched,
                     "lora_cfg": lora_settings(), "kd_cfg": kd_settings(), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, latest_meta_path)

    def validate(where):
        """validation loss of the EMA weights; saves student_best.pt when it is the best so far"""
        nonlocal best_val
        with ema_ctx():
            if teacher is not None:
                full = eval_loss(model, val_s, device, teacher=teacher, full=True)
                vl = full["ce"]
                val_extra.update(val_kd=full["kl"], val_top1_agree=full["agree"])
                log(f"  validation ({where}): CE {vl:.4f} | distillation term {full['kl']:.4f} | "
                    f"top-1 agreement with the teacher {full['agree']:.3f}")
            else:
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

    stopped, n_skipped, steps_this_run, run, run_n, run_ce, run_kl = False, 0, 0, 0.0, 0, 0.0, 0.0
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
            step_loss = step_ce = step_kl = 0.0
            for m in range(0, len(idxs), MICRO_BS):
                x, y = collate([train_s[i] for i in idxs[m:m + MICRO_BS]], model.cfg.pad_id)
                loss, _, info = summary_loss_sum(model, x.to(device), y.to(device), teacher)
                loss = loss / n_tgt                      # token-level mean over the whole effective batch
                scaler.scale(loss).backward()
                step_loss += loss.item()
                step_ce += info["ce"] / n_tgt
                step_kl += info["kl"] / n_tgt
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            if math.isfinite(step_loss) and bool(torch.isfinite(gnorm)):
                scaler.step(opt)
                if ema is not None:
                    ema.update(state["global_step"])
                state["ep_loss"] += step_loss
                state["ep_ce"] += step_ce
                state["ep_n"] += 1
                run += step_loss
                run_ce += step_ce
                run_kl += step_kl
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
                kd_txt = f", kd {run_kl / max(run_n, 1):.4f}" if teacher is not None else ""
                log(f"[ep {epoch + 1}/{EPOCHS}] step {step + 1}/{steps_per_epoch} (global {state['global_step']}) | "
                    f"loss {run / max(run_n, 1):.4f} (ce {run_ce / max(run_n, 1):.4f}{kd_txt}) | gnorm {float(gnorm):.2f} | lr x{mult:.3f} | {sps:.2f}s/step | "
                    f"epoch ETA {fmt_time((steps_per_epoch - step - 1) * sps)} | all-epochs ETA "
                    f"{fmt_time((total_steps - state['global_step']) * sps)} | train time {fmt_time(state['train_sec'])}")
                run, run_n, run_ce, run_kl = 0.0, 0, 0.0, 0.0
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
        tr_ce = state["ep_ce"] / max(state["ep_n"], 1)
        state.update(epoch=epoch + 1, bi=0, ep_loss=0.0, ep_n=0, ep_ce=0.0)
        vl, improved = validate(f"epoch{epoch + 1}")
        history.append({"epoch": epoch + 1, "train_loss": tr, "train_ce": tr_ce, "val_loss": vl, "val_ppl": math.exp(min(vl, 20)),
                        **val_extra,
                        "global_step": state["global_step"], "train_hours": state["train_sec"] / 3600, "best": improved})
        atomic_json(history, f"{CKPT_DIR}/train_history.json")
        save_latest({"loss": vl, "global_step": state["global_step"]})
        tick()
        log(f"===== epoch {epoch + 1}/{EPOCHS} finished | train loss {tr:.4f} (ce {tr_ce:.4f}) | val CE {vl:.4f} (ppl {math.exp(min(vl, 20)):.2f}) | "
            f"{'NEW BEST, saved student_best.pt' if improved else 'best %.4f' % best_val} | train time {fmt_time(state['train_sec'])}")
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
            "steps_per_epoch": steps_per_epoch, "baseline": baseline, "epochs_done": state["epoch"]}


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
    ban = torch.tensor(STATE["ban_list"], dtype=torch.long, device=dev)
    seq = torch.full((B, max_new), cfg.pad_id, dtype=torch.long, device=dev)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    with amp_ctx():
        cache, logits, lengths, valid = start_generation(model, ids, mask, max_new)
        for t in range(max_new):
            logits = logits.float()
            logits.index_fill_(1, ban, float("-inf"))      # specials (except EOS = the end marker) + padding rows
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


# ---------------------------- output post-processing ----------------------------
# The Phase-1 word-level tokenizer lower-cases everything and has no whitespace tokens, so a decoded chunk summary looks like
#   "the high court held that the appellant 's detention under section 3 of the act was valid"
# and - because its WORD_PATTERN contains the character-class range `\a-z` (BEL..'z', which swallows every ASCII punctuation
# mark) - it contains NO punctuation at all (init_state detects this: STATE['punct']).  detok() re-attaches the apostrophes
# (and the punctuation, if a tokenizer does produce it), restore_case() brings back capitals (word forms learnt from the
# TRAINING summaries) and render_chunk() closes every chunk summary with a full stop.
# ROUGE / BERTScore of the submission are computed on this final text.
ABBREV = frozenset("no nos s ss sec secs art arts cl ch sub v vs dr mr mrs ms hon rs ltd pvt co anr ors st u m w r ibid etc viz "
                   "para paras vol ed".split())
LOWER_WORD = re.compile(r"[a-z]+")


def _fix_quotes(t):
    parts = t.split('"')
    if len(parts) == 1:
        return t
    res = parts[0]
    for i, p in enumerate(parts[1:]):
        if i % 2 == 0:                                                    # opening quote: glue to the next word
            res = res.rstrip() + ' "' + p.lstrip()
        else:                                                             # closing quote: glue to the previous word
            res = res.rstrip() + '"' + (" " if p[:1].isspace() else "") + p.lstrip()
    return res


def detok(text):
    """undo the word-by-word spacing of the word-level tokenizer"""
    t = " ".join(str(text).split())
    t = _fix_quotes(t)
    t = re.sub(r"\s+([.,;:!?%)\]}])", r"\1", t)
    t = re.sub(r"([(\[{])\s+", r"\1", t)
    t = re.sub(r"\s+('(?:s|t|re|ve|m|ll|d)\b)", r"\1", t)
    t = re.sub(r"\s*’\s*(?=(?:s|t|re|ve|m|ll|d)\b)", "’", t)
    t = re.sub(r"(\w)\s+-\s+(\w)", r"\1-\2", t)
    t = re.sub(r"(?<=\d)\s*([.,/:])\s*(?=\d)", r"\1", t)
    return " ".join(t.split())


def split_output_sentences(text):
    """sentence splitter for GENERATED text (works without capital letters; knows the common legal abbreviations)"""
    out, cur = [], []
    for w in str(text).split():
        cur.append(w)
        core = w.rstrip("\"')]")
        boundary = bool(core) and core[-1] in ".!?"
        if boundary and core[-1] == ".":
            stem = re.sub(r"[^a-z]", "", core.lower())
            if stem in ABBREV or len(stem) == 1:
                boundary = False
        if boundary or len(cur) >= MAX_SENT_WORDS:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def build_case_map(recs, max_docs=4000, min_count=3, min_share=0.6):
    """lower-case word -> its usual written form in the reference summaries (e.g. ipc -> IPC, supreme -> Supreme).
    Sentence-initial words are skipped (their capital is positional); only forms used >= min_share of the time are kept."""
    forms = {}
    for r in recs[:max_docs]:
        t = r["summary"]
        for m in re.finditer(r"[A-Za-z]+", t):
            j = m.start() - 1
            while j >= 0 and t[j] in " \"'([":
                j -= 1
            if j < 0 or t[j] in ".!?:":
                continue
            w = m.group(0)
            forms.setdefault(w.lower(), Counter())[w] += 1
    cmap = {}
    for lw, c in forms.items():
        w, n = c.most_common(1)[0]
        if w != lw and sum(c.values()) >= min_count and n / sum(c.values()) >= min_share:
            cmap[lw] = w
    log(f"case map: {len(cmap):,} words with a usual non-lower-case form, e.g. {list(cmap.items())[:6]}")
    return cmap


def restore_case(s):
    cm = STATE.get("case_map")
    if cm:
        s = LOWER_WORD.sub(lambda m: cm.get(m.group(0), m.group(0)), s)
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
        if ch.isdigit():
            break
    return s


def output_units(text):
    """units a generated chunk summary is deduplicated / trimmed in: sentences, or - when the tokenizer cannot produce any
    punctuation (STATE['punct'] False, see init_state) - consecutive windows of GEN_WINDOW_WORDS words"""
    post = STATE.get("postprocess", False)
    text = detok(text) if post else str(text)
    if STATE.get("punct", True):
        return split_output_sentences(text)
    w = text.split()
    return [" ".join(w[i:i + GEN_WINDOW_WORDS]) for i in range(0, len(w), GEN_WINDOW_WORDS)]


def render_chunk(units):
    post = STATE.get("postprocess", False)
    if not units:
        return ""
    if STATE.get("punct", True):
        return " ".join(restore_case(s) if post else s for s in units)
    txt = " ".join(units)                                   # no punctuation available: one 'sentence' per chunk summary
    txt = restore_case(txt) if post else txt
    return txt if END_PUNCT.search(txt) else txt + "."


def assemble_summary(parts, doc_words):
    """parts: [(chunk_summary_text, hit_length_limit)] in document order -> final summary.
    Drops a cut-off last unit, near-duplicate units and, if needed, trims proportionally to a length cap."""
    kept_sets, per_chunk = [], []
    for text, truncated in parts:
        sents = output_units(text)
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
    return " ".join(r for r in (render_chunk(ks) for ks in per_chunk) if r)


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
# the probe prompts are LONG on purpose: Model 4 reads earlier 128-token chunks through its memory layers, so the checks below only
# mean something when the prompts cross several chunk borders (and one is shorter than the other -> the decoder has to left-pad it)
PROBE_TEXTS = ("The appellant filed a writ petition before the High Court challenging the order of detention passed under the "
               "Act, and the Court held that the detention was valid. " * 14,
               "Held: the appeal is allowed. " * 30)


def probe_prompts(tok):
    return [build_prompt(tok, 0, 1, tok.encode(t)[:CHUNK_TOKENS]) for t in PROBE_TEXTS]


@torch.no_grad()
def probe_features(model, tok):
    """next-token features (head 1) of the probe prompts, fp32, eval mode - a fingerprint of the network function"""
    dev = next(model.parameters()).device
    was = model.training
    model.eval()
    try:
        return [lm_features(model, torch.tensor([p.tolist()], device=dev)).float().cpu() for p in probe_prompts(tok)]
    finally:
        model.train(was)


@torch.no_grad()
def check_block_attention():
    """block-wise sliding-window attention == plain masked attention (otherwise the slower masked version is used)"""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for window in (WINDOW_SMALL, WINDOW_LARGE):
        for T in (window - 5, 300, 640):
            q, k, v = (torch.randn(2, 4, T, 16, generator=g).to(dev) for _ in range(3))
            worst = max(worst, (local_attention(q, k, v, window) - local_attention_ref(q, k, v, window)).abs().max().item())
    if worst > 1e-4:
        LOCAL_IMPL["mode"] = "mask"
        log(f"WARNING: block-local attention differs from the masked reference by {worst:.2e} -> using masked SDPA instead")
    else:
        log(f"check OK: block-local attention == masked reference (max diff {worst:.2e})")


@torch.no_grad()
def check_causality(model, tok):
    """the student is causal: changing the tokens from position p on never changes the logits before p - also across the
    128-token chunk borders of the memory layers (fp32)"""
    dev = next(model.parameters()).device
    was, amp_was = model.training, AMP["enabled"]
    AMP["enabled"] = False
    model.eval()
    try:
        ia, ib = tok.encode(PROBE_TEXTS[0]), tok.encode(PROBE_TEXTS[1])
        a = torch.tensor([(ia * 3)[:700]], device=dev)
        b = torch.tensor([(ib * 8)[:700]], device=dev)
        assert a.size(1) == 700 and b.size(1) == 700, "probe texts too short for the causality check"
        base = model(a)
        for p in (131, 384, 520):            # inside a chunk / exactly on a chunk border / inside a later chunk
            out = model(torch.cat([a[:, :p], b[:, p:]], dim=1))
            past = (out[:, :p] - base[:, :p]).abs().max().item()
            fut = (out[:, p:] - base[:, p:]).abs().max().item()
            if past > 1e-3:
                raise RuntimeError(f"causality check failed: changing tokens >= {p} moved earlier logits by {past:.2e}")
            if fut < 1e-3:
                raise RuntimeError(f"causality check is blind: changing tokens >= {p} barely moves later logits ({fut:.2e})")
    finally:
        AMP["enabled"] = amp_was
        model.train(was)
    log("check OK: the student is causal (past logits unchanged when later tokens change, also across memory-chunk borders)")


def check_identity_at_init(model, tok, before):
    """a freshly initialised LoRA adapter must be the identity: the network function has to be unchanged"""
    after = probe_features(model, tok)
    diff = max((a - b).abs().max().item() for a, b in zip(before, after))
    log(f"identity check: max |features(student + fresh adapter) - features(Phase-1 Model 4)| = {diff:.2e}")
    if diff > 1e-4:
        raise RuntimeError(f"a fresh LoRA adapter changes the network function (diff {diff:.2e}) - init is wrong")


@torch.no_grad()
def check_cached_decoding(model, tok, n_new=8):
    """KV-cached batched decoding (left padding, RoPE positions, sliding windows, memory layers) must produce the same greedy
    tokens as plain forward passes over the growing sequence. Runs in fp32 BEFORE training, so a decoding bug is found in
    seconds, not after 9 hours."""
    prompts = probe_prompts(tok)
    dev = next(model.parameters()).device
    ban = torch.tensor(STATE["ban_list"], dtype=torch.long, device=dev)
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
                lg[ban] = float("-inf")
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
    log(f"decoding check OK: cached batched greedy decoding == plain forward passes ({n_new} tokens, 2 prompts of "
        f"{'/'.join(str(len(p)) for p in prompts)} tokens, memory chunk = {MEM_CHUNK_SIZE})")


@torch.no_grad()
def check_kd_loss():
    """numerical check of the distillation term against a naive per-position computation (random logits, milliseconds)"""
    g = torch.Generator().manual_seed(3)
    V, N, T, eos = 40, 9, 2.5, 2
    s, t = torch.randn(N, V, generator=g) * 2, torch.randn(N, V, generator=g) * 2
    y = torch.randint(3, V, (N,), generator=g)
    y[0] = eos
    for eos_ok in (True, False):
        ce, kl, agree = kd_loss_parts(s, t, y, T, eos, eos_ok)
        ref_ce = sum(-torch.log_softmax(s[i], -1)[y[i]] for i in range(N))
        ref_kl = 0.0
        for i in range(N):
            if not eos_ok and int(y[i]) == eos:
                continue
            si, ti = s[i], t[i]
            if not eos_ok:
                keep = torch.arange(V) != eos
                si, ti = si[keep], ti[keep]
            p, q = torch.softmax(ti / T, -1), torch.softmax(si / T, -1)
            ref_kl += float((p * (p.log() - q.log())).sum()) * T * T
        ref_agree = int((s.argmax(-1) == t.argmax(-1)).sum())
        if abs(float(ce) - float(ref_ce)) > 1e-3 or abs(float(kl) - ref_kl) > 1e-3 or agree != ref_agree:
            raise RuntimeError(f"KD loss self-check failed (eos_ok={eos_ok}): ce {float(ce):.5f}/{float(ref_ce):.5f} "
                               f"kl {float(kl):.5f}/{ref_kl:.5f} agree {agree}/{ref_agree}")
    _, same, _ = kd_loss_parts(s, s, y, T, eos, True)
    if abs(float(same)) > 1e-4:
        raise RuntimeError(f"KD loss self-check failed: KL(student || student) = {float(same):.2e} (must be 0)")
    log("check OK: distillation loss == naive reference (T^2-scaled KL, EOS handling), KL(p||p) = 0")


@torch.no_grad()
def merge_with_check(model, tok):
    """fold LoRA into the base weights and verify that the network function (next-token logits) did not change"""
    dev = next(model.parameters()).device
    xs = [torch.tensor([p.tolist()], device=dev) for p in probe_prompts(tok)]
    model.eval()
    before = [model(x)[0, -1].float() for x in xs]
    merge_lora(model)
    after = [model(x)[0, -1].float() for x in xs]
    diff = max((a - b).abs().max().item() for a, b in zip(before, after))
    log(f"merge check: max |logits(LoRA) - logits(merged)| = {diff:.2e}")
    if diff > 5e-3:
        msg = f"merged model differs from the LoRA model (max logit diff {diff:.2e})"
        if SMOKE_TEST:
            raise RuntimeError(msg)
        log("  WARNING: " + msg)
    return diff


# =====================================================================================
# MAIN
# =====================================================================================
def teacher_check(teacher, info, val_s, device):
    """the rebuilt teacher must reproduce the validation CE stored with its adapter - otherwise its soft targets are garbage
    (wrong tokenizer, changed chunking constants, adapter of another model)"""
    tvl = eval_loss(teacher, val_s, device)
    stored = info.get("adapter_val_loss", float("nan"))
    log(f"teacher ({info['kind']}) validation CE on the summary tokens: {tvl:.4f}"
        + (f" | stored with its adapter: {stored:.4f}" if math.isfinite(stored) else ""))
    if info["kind"] == "finetuned" and not SMOKE_TEST and math.isfinite(stored):
        diff = abs(tvl - stored)
        ok = diff <= TEACHER_CHECK_TOL
        log(f"teacher check: |{tvl:.4f} - {stored:.4f}| = {diff:.4f} -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            msg = (f"the rebuilt teacher does not reproduce the validation loss stored in its adapter ({tvl:.4f} vs {stored:.4f}): "
                   f"the teacher would give wrong soft targets. Check that the SAME tokenizer, chunking constants (CHUNK_TOKENS, "
                   f"TGT_MAX_TOKENS, KEEP_EMPTY_P, ...), VAL_DOCS and SEED were used as in the Phase-6 Model-3 run "
                   f"(or set TEACHER_STRICT = False if you changed them on purpose).")
            if TEACHER_STRICT:
                raise RuntimeError(msg)
            log("  WARNING: " + msg)
    return tvl


def teacher_generation_metrics(info):
    """validation ROUGE / BERTScore of the teacher, as written by its own Phase-6 run (for the comparison in the report)"""
    if not info or not info.get("path"):
        return None
    return read_json(os.path.join(os.path.dirname(info["path"]), "val_generation_metrics.json"))


def find_adapter():
    p = os.path.join(CKPT_DIR, "student_best.pt")
    if os.path.exists(p):
        return p
    hits = [d for d in walk_find("student_best.pt", [INPUT_ROOT])
            if (read_json(os.path.join(d, "student_best_meta.json")) or {}).get("tag") == RUN_TAG]
    if hits:
        return os.path.join(hits[0], "student_best.pt")
    raise FileNotFoundError("student_best.pt not found - train first or attach the earlier run's output")


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
    log(f"run tag {RUN_TAG} | " + (f"distillation Model 3 -> Model 4: alpha {KD_ALPHA}, T {KD_TEMP}" if USE_TEACHER
                                   else "NO distillation (plain LoRA baseline)")
        + f" | smoke test: {SMOKE_TEST} | epochs {EPOCHS} | effective batch {EFF_BS} chunks | work dir {WORK_DIR}")

    if train_raw is None:
        train_raw, test_raw = load_records("train"), load_records("test")
    student, scfg = load_student(device)
    teacher = None
    if USE_TEACHER:
        teacher, tcfg = load_teacher_base(device)
        if tcfg.real_vocab != scfg.vocab_size:
            raise RuntimeError(f"teacher vocabulary ({tcfg.real_vocab}) != student vocabulary ({scfg.vocab_size}): the two models must "
                               f"share the same 50k BPE")
    recs = clean_train_records(train_raw)
    random.Random(SEED).shuffle(recs)
    longest = sorted(recs[:60], key=lambda r: -len(r["document"]))[:6]
    checks = [("student (Model 4)", student, student.pretrain_val_loss)]
    if teacher is not None:
        checks.append(("teacher (Model 3)", teacher, teacher.pretrain_val_loss))
    tok = choose_tokenizer(checks, scfg.vocab_size, [r["document"] for r in longest])
    init_state(tok, scfg.vocab_size)
    teacher_info = None
    if teacher is not None:
        teacher_info = attach_teacher_adapter(teacher)   # after the tokenizer check: that one is made with the pretrained Model 3
        STATE["teacher_eos_ok"] = teacher_info["eos_ok"]
    check_kd_loss()
    check_block_attention()
    check_causality(student, tok)                        # fail fast: before any training time is spent
    check_cached_decoding(student, tok)

    if SMOKE_TEST:
        val_recs = smoke_subset(recs, N_SMOKE["val"])
        used = {r["id"] for r in val_recs}
        train_recs = smoke_subset([r for r in recs if r["id"] not in used], N_SMOKE["train"])
    else:
        val_recs, train_recs = recs[:VAL_DOCS], recs[VAL_DOCS:]
    log(f"documents: {len(train_recs):,} train / {len(val_recs)} validation")
    STATE["postprocess"] = bool(getattr(tok, "needs_detok", False))
    STATE["case_map"] = build_case_map(train_recs) if STATE["postprocess"] else {}

    def adapt(m):
        """wrap the frozen Phase-1 student with a FRESH LoRA adapter and prove that it starts as the identity"""
        before = probe_features(m, tok)
        apply_lora(m, lora_settings())
        check_identity_at_init(m, tok, before)

    result, teacher_val = None, None
    if RUN_TRAIN:
        train_s, val_s = prepare_samples(tok, train_recs, val_recs)
        if MAX_TRAIN_CHUNKS and len(train_s) > MAX_TRAIN_CHUNKS:
            train_s = random.Random(SEED).sample(train_s, MAX_TRAIN_CHUNKS)
            log(f"training set capped to {len(train_s):,} chunks")
        if teacher is not None:
            teacher_val = teacher_check(teacher, teacher_info, val_s, device)
        adapt(student)
        if SMOKE_TEST:
            spe = max(1, len(train_s) // EFF_BS)
            stop = min(SMOKE_STOP_STEP, max(1, spe // 2))
            log(f"\n===== SMOKE: training session 1 - cut after {stop} of {EPOCHS * spe} steps ({spe} steps/epoch) =====")
            r1 = train(student, teacher, train_s, val_s, device, t_start, stop_at_step=stop)
            assert not r1["finished"] and r1["final_step"] == stop, f"session 1 should stop at step {stop}: {r1}"
            log("\n===== SMOKE: training session 2 - fresh student, resumes from the checkpoint =====")
            student, scfg = load_student(device)
            adapt(student)
            result = train(student, teacher, train_s, val_s, device, t_start)
            assert result["resumed_from"] is not None and result["resumed_from"][2] == r1["final_step"], \
                f"session 2 did not resume from session 1's checkpoint: {result['resumed_from']}"
            assert result["finished"] and result["final_step"] == EPOCHS * spe, \
                f"expected {EPOCHS * spe} steps in total, got {result['final_step']}"
            log("SMOKE: interrupted + resumed training OK "
                f"(resumed at epoch {result['resumed_from'][0] + 1}, batch {result['resumed_from'][1]}, step {result['resumed_from'][2]})")
        else:
            result = train(student, teacher, train_s, val_s, device, t_start)
        ck = torch.load(os.path.join(CKPT_DIR, "student_best.pt"), map_location="cpu", weights_only=False)
        load_adapter_state(student, ck["lora"])
        log(f"best adapter restored (val CE {ck['val_loss']:.4f}, after {ck['epochs_done']} epochs + {ck['batch']} batches, "
            f"step {ck['global_step']})")
    else:
        ck = torch.load(find_adapter(), map_location="cpu", weights_only=False)
        apply_lora(student, ck["lora_cfg"])
        load_adapter_state(student, ck["lora"])

    rep = lora_report(student)
    if rep is not None:
        atomic_json(rep, os.path.join(CKPT_DIR, "lora_report.json"))
    merge_with_check(student, tok)
    student.eval()
    if SAVE_MERGED:
        atomic_torch_save({"model": student.state_dict(), "config": asdict(scfg), "tag": STUDENT_TAG}, os.path.join(CKPT_DIR, "merged_model.pt"))

    gen_res = None
    if RUN_VAL_GEN and val_recs:
        gen_res, _ = eval_generation(student, tok, val_recs[:VAL_GEN_DOCS], device, "val")
        gen_res.update(use_teacher=USE_TEACHER, kd_alpha=KD_ALPHA, kd_temp=KD_TEMP)
        atomic_json(gen_res, os.path.join(CKPT_DIR, "val_generation_metrics.json"))

    atomic_json({"run_tag": RUN_TAG, "use_teacher": USE_TEACHER, "kd_alpha": KD_ALPHA, "kd_temp": KD_TEMP, "teacher": teacher_info,
                 "teacher_val_ce": teacher_val, "teacher_val_generation": teacher_generation_metrics(teacher_info),
                 "student_val_ce_before_adaptation": None if result is None else result.get("baseline"),
                 "student_best_val_ce": None if result is None else result["best_val"],
                 "epochs_done": None if result is None else result.get("epochs_done"),
                 "student_val_generation": gen_res, "lora": lora_settings(), "smoke_test": SMOKE_TEST},
                os.path.join(CKPT_DIR, "kd_report.json"))

    if RUN_PREDICT and test_raw is not None:
        test_recs = [{"id": r["id"], "document": " ".join(str(r["document"]).split())} for r in test_raw]
        if SMOKE_TEST:
            test_recs = smoke_subset(test_recs, N_SMOKE["test"])
        summaries = summarize_docs(student, tok, [(r["id"], r["document"]) for r in test_recs], desc="test")
        write_submission(test_recs, summaries)
        if SMOKE_TEST:
            preds = read_json(os.path.join(OUT_DIR, "prediction.json"))
            assert preds and len(preds) == len(test_recs) and all(p["summary"].strip() for p in preds)

    if SMOKE_TEST:
        log("\n" + "=" * 90 + "\nSMOKE TEST PASSED: tokenizer check (both models), teacher, KD-loss / attention / causality / decoding checks, "
            "LoRA identity-at-init, interrupted + resumed distillation training, best adapter, LoRA merge, validation generation, "
            "test prediction and submission files all work.\nNow set SMOKE_TEST = False and run for real.\n" + "=" * 90)
    elif result is not None and not result["finished"]:
        log("\nTRAINING IS NOT FINISHED (session time budget). A submission was written from the best adapter so far.\n"
            "To continue: Save & Commit this notebook, attach its output as an input of the next session (same RUN_TAG, same other "
            "inputs), run again - it resumes at the saved epoch/batch.")
    log(f"done in {fmt_time(time.time() - t_start)}")


if __name__ == "__main__":
    main()
