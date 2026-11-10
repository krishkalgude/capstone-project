# Capstone Project: Phase 6

Phase 6 carries on with the fine-tuning, with LoRA, rsLoRA and LoRMA runs on models 1, 2 (preprocessed) and 3, plus a distillation run from model 3 to model 4. One difference from the earlier phases: everything here is a plain Python script (`.py`), not a notebook.

## Naming

Every file starts with `p6_`. After that:

- `m1`, `m2p`, `m3` are models 1, 2 (preprocessed) and 3 from Phase 1. Model 1 is the encoder-decoder. `m2p` is model 2 trained on the preprocessed 32k BPE tokenizer, so it's the "clean" version of model 2. The `p` is for preprocessed.
- `kd_3_to_4` is knowledge distillation from model 3 to model 4.
- The suffix (`hier`, `chunked`, `docpool`) is the variant of that run. Several scripts differ only in this part.

## What's in here

| Script | Model | Method | Variant |
|---|---|---|---|
| `p6_m1_lora_hier.py` | Model 1 | LoRA | hier |
| `p6_m2p_lora_hier.py` | Model 2 (preprocessed) | LoRA | hier |
| `p6_m2p_lora_chunked.py` | Model 2 (preprocessed) | LoRA | chunked |
| `p6_m2p_lora_docpool.py` | Model 2 (preprocessed) | LoRA | docpool |
| `p6_m2p_lorma_hier.py` | Model 2 (preprocessed) | LoRMA | hier |
| `p6_m2p_rslora_hier.py` | Model 2 (preprocessed) | rsLoRA | hier |
| `p6_m3_lorma.py` | Model 3 | LoRMA | |
| `p6_kd_3_to_4_lora.py` | Model 3 to Model 4 | Knowledge distillation + LoRA | |

## Before you run anything

Each script needs two things from Phase 1:

1. **The tokenizer output** from the matching BPE notebook.
2. **The pretrained model weights** that came out of the matching Phase 1 model notebook.

Fine-tuning starts from those weights, and the model only makes sense with the exact tokenizer it was pretrained with, so the pairing matters.

| Phase 6 script | Tokenizer (from Phase 1) | Pretrained weights (from Phase 1) |
|---|---|---|
| `p6_m1_lora_hier` | `BPE_model_1_2.ipynb` (35k) | `model1_ed_t.ipynb` |
| all `p6_m2p_*` scripts | `bpe_final_model2.ipynb` (32k) | `Model2_clean_BPE_final_gelu.ipynb` |
| `p6_m3_lorma` | `legal_bpe_50k.ipynb` (50k) | `model3_postln_wnorm.ipynb` |
| `p6_kd_3_to_4_lora` | `legal_bpe_50k.ipynb` (50k) | `model3_postln_wnorm.ipynb` and `model4.ipynb` |

The distillation script uses two models, so it needs the weights for both the teacher and the student.

The three tokenizers (35k, 32k and 50k) give different token ids, so they aren't interchangeable. Use the tokenizer that matches the model, otherwise the vocab won't line up and you'll get errors or nonsense output.

## Running a script

1. Run the tokenizer notebook from Phase 1 for your model (or reuse its saved output).
2. Get the pretrained weights from the matching Phase 1 model notebook.
3. Point the script at both (check the top of the script for where it expects them), then run it with `python`.
