# Capstone Project: Phase 2

Phase 2 is where the fine-tuning starts. Every notebook here takes a model pretrained in Phase 1 (see the `phase1` branch) and fine-tunes it with a different method: LoRA, full fine-tuning, prefix fine-tuning, LoRMA and knowledge distillation.

## Naming

The notebooks all start with `p2_`. After that:

- `m1`, `m2`, `m3`, `m4` are models 1 to 4 from Phase 1.
- `m2p` is model 2 trained on the preprocessed BPE tokenizer (the 32k one), which is the "clean" version of model 2 from Phase 1. The `p` is for preprocessed.

## What's in here

| Notebook | Model | Method |
|---|---|---|
| `p2_m2_LoRA.ipynb` | Model 2 | LoRA |
| `p2_m2p_LoRA.ipynb` | Model 2 (preprocessed) | LoRA |
| `p2_m2p_FullFT.ipynb` | Model 2 (preprocessed) | Full fine-tuning |
| `p2_m2p_Prefix_FT.ipynb` | Model 2 (preprocessed) | Prefix fine-tuning |
| `p2_model3_LoRA.ipynb` | Model 3 | LoRA |
| `p2_m3_LoRMA.ipynb` | Model 3 | LoRMA |
| `p2_KD_3_to_4.ipynb` | Model 3 to Model 4 | Knowledge distillation |
| `p2_knowledge-distillation_finetune_v2.ipynb` | | Knowledge distillation, second version |

## Before you run anything

Each notebook needs two things from Phase 1:

1. **The tokenizer output** from the matching BPE notebook.
2. **The pretrained model weights** that came out of the matching Phase 1 model notebook.

Fine-tuning starts from those weights, and the model only makes sense with the exact tokenizer it was pretrained with, so the pairing matters.

| Phase 2 notebook | Tokenizer (from Phase 1) | Pretrained weights (from Phase 1) |
|---|---|---|
| `p2_m2_LoRA` | `BPE_model_1_2.ipynb` (35k) | `model2_decoder_only_final.ipynb` |
| `p2_m2p_LoRA`, `p2_m2p_FullFT`, `p2_m2p_Prefix_FT` | `bpe_final_model2.ipynb` (32k) | `Model2_clean_BPE_final_gelu.ipynb` |
| `p2_model3_LoRA`, `p2_m3_LoRMA` | `legal_bpe_50k.ipynb` (50k) | `model3_postln_wnorm.ipynb` |
| `p2_KD_3_to_4` | `legal_bpe_50k.ipynb` (50k) | `model3_postln_wnorm.ipynb` and `model4.ipynb` |

The distillation notebooks use two models, so they need the tokenizer and weights for both the teacher and the student.

One thing that's easy to mix up: `m2` and `m2p` are both "model 2", but they were pretrained with different tokenizers (35k vs 32k). Use the tokenizer and weights that match the notebook you're running, otherwise the vocab won't line up and you'll get errors or nonsense output.

## Running a notebook

1. Run the tokenizer notebook from Phase 1 for your model (or reuse its saved output).
2. Get the pretrained weights from the matching Phase 1 model notebook.
3. Point the Phase 2 notebook at both, then run it.
