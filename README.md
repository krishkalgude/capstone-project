# Capstone Project: Phase 4

Phase 4 carries on with the fine-tuning. The notebooks here fine-tune the Phase 1 models with LoRA, rsLoRA and LoRMA, bring in the Longformer, and include another knowledge distillation run.

## Naming

Most notebooks start with `p4_`. One of them starts with `phase4_` instead, which is the same phase, just named differently. After the prefix:

- `m2p`, `m3` are models 2 (preprocessed) and 3 from Phase 1. `m2p` is model 2 trained on the preprocessed 32k BPE tokenizer, so it's the "clean" version of model 2. The `p` is for preprocessed.
- `longformer` is the Longformer from Phase 1.
- `KD_3_to_4` is knowledge distillation from model 3 to model 4.

## What's in here

| Notebook | Model | Method |
|---|---|---|
| `longformer_model.ipynb` | Longformer | |
| `p4_longformer_lora.ipynb` | Longformer | LoRA |
| `phase4_m2p_LoRA.ipynb` | Model 2 (preprocessed) | LoRA |
| `p4_m2p_rslora.ipynb` | Model 2 (preprocessed) | rsLoRA |
| `p4_m3_lorma.ipynb` | Model 3 | LoRMA |
| `p4_KD_3_to_4.ipynb` | Model 3 to Model 4 | Knowledge distillation |

## Before you run anything

Each notebook needs two things from Phase 1:

1. **The tokenizer output** from the matching BPE notebook.
2. **The pretrained model weights** that came out of the matching Phase 1 model notebook.

Fine-tuning starts from those weights, and the model only makes sense with the exact tokenizer it was pretrained with, so the pairing matters.

| Phase 4 notebook | Tokenizer (from Phase 1) | Pretrained weights (from Phase 1) |
|---|---|---|
| `longformer_model`, `p4_longformer_lora` | `bpe_final_model2.ipynb` (32k) | `longformer-f.ipynb` |
| `phase4_m2p_LoRA`, `p4_m2p_rslora` | `bpe_final_model2.ipynb` (32k) | `Model2_clean_BPE_final_gelu.ipynb` |
| `p4_m3_lorma` | `legal_bpe_50k.ipynb` (50k) | `model3_postln_wnorm.ipynb` |
| `p4_KD_3_to_4` | `legal_bpe_50k.ipynb` (50k) | `model3_postln_wnorm.ipynb` and `model4.ipynb` |

The distillation notebook uses two models, so it needs the weights for both the teacher and the student.

The three tokenizers (35k, 32k and 50k) give different token ids, so they aren't interchangeable. Use the tokenizer that matches the model, otherwise the vocab won't line up and you'll get errors or nonsense output.

## Running a notebook

1. Run the tokenizer notebook from Phase 1 for your model (or reuse its saved output).
2. Get the pretrained weights from the matching Phase 1 model notebook.
3. Point the Phase 4 notebook at both, then run it.
