# Capstone Project: Phase 3

Phase 3 continues the fine-tuning work. Like Phase 2, every notebook here starts from a model pretrained in Phase 1 (see the `phase1` branch) and fine-tunes it, this time with LoRA and a few of its variants (LoRMA, AdaLoRA, rsLoRA), plus another knowledge distillation run.

## Naming

The notebooks all start with `p3_`. After that:

- `m2`, `m3`, `m4` are models 2, 3 and 4 from Phase 1.
- `m2p` is model 2 trained on the preprocessed BPE tokenizer (the 32k one), which is the "clean" version of model 2 from Phase 1. The `p` is for preprocessed.

## What's in here

| Notebook | Model | Method |
|---|---|---|
| `p3_m2_LoRA.ipynb` | Model 2 | LoRA |
| `p3_m2_LoRMA.ipynb` | Model 2 | LoRMA |
| `p3_m2p_AdaLoRA.ipynb` | Model 2 (preprocessed) | AdaLoRA |
| `p3_m2p_rsLoRA.ipynb` | Model 2 (preprocessed) | rsLoRA |
| `p3_KD_m3_to_4.ipynb` | Model 3 to Model 4 | Knowledge distillation |

## Before you run anything

Each notebook needs two things from Phase 1:

1. **The tokenizer output** from the matching BPE notebook.
2. **The pretrained model weights** that came out of the matching Phase 1 model notebook.

Fine-tuning starts from those weights, and the model only makes sense with the exact tokenizer it was pretrained with, so the pairing matters.

| Phase 3 notebook | Tokenizer (from Phase 1) | Pretrained weights (from Phase 1) |
|---|---|---|
| `p3_m2_LoRA`, `p3_m2_LoRMA` | `BPE_model_1_2.ipynb` (35k) | `model2_decoder_only_final.ipynb` |
| `p3_m2p_AdaLoRA`, `p3_m2p_rsLoRA` | `bpe_final_model2.ipynb` (32k) | `Model2_clean_BPE_final_gelu.ipynb` |
| `p3_KD_m3_to_4` | `legal_bpe_50k.ipynb` (50k) | `model3_postln_wnorm.ipynb` and `model4.ipynb` |

The distillation notebook uses two models, so it needs the weights for both the teacher and the student.

As in Phase 2, `m2` and `m2p` are both "model 2" but were pretrained with different tokenizers (35k vs 32k). Use the tokenizer and weights that match the notebook you're running, otherwise the vocab won't line up and you'll get errors or nonsense output.

## Running a notebook

1. Run the tokenizer notebook from Phase 1 for your model (or reuse its saved output).
2. Get the pretrained weights from the matching Phase 1 model notebook.
3. Point the Phase 3 notebook at both, then run it.
