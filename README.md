# Capstone Project: Phase 7

This is the last phase. It has a final round of fine-tuning runs (LoRA, LoRMA and LoRMA+ on models 1, 2 (preprocessed) and 3), the knowledge distillation runs from model 3 to model 4, and two notebooks that re-score runs saved earlier.

## Naming

Every notebook starts with `p7_`. After that:

- `m1`, `m2p`, `m3` are models 1, 2 (preprocessed) and 3 from Phase 1. Model 1 is the encoder-decoder. `m2p` is model 2 trained on the preprocessed 32k BPE tokenizer, so it's the "clean" version of model 2. The `p` is for preprocessed.
- `KD_3_to_4` is knowledge distillation from model 3 to model 4. In `kd_combined_m3teacher_m4student`, model 3 is the teacher and model 4 is the student.
- `rescore_saved_runs` goes back over runs that were saved earlier. `_v2` is the second version.

## What's in here

| Notebook | Model | Method |
|---|---|---|
| `p7_m1_lorma.ipynb` | Model 1 | LoRMA |
| `p7_m2p_lora.ipynb` | Model 2 (preprocessed) | LoRA |
| `p7_m2p_lorma.ipynb` | Model 2 (preprocessed) | LoRMA |
| `p7_m3_lorma+.ipynb` | Model 3 | LoRMA+ |
| `p7_KD_3_to_4.ipynb` | Model 3 to Model 4 | Knowledge distillation |
| `p7_kd_combined_m3teacher_m4student.ipynb` | Model 3 (teacher) to Model 4 (student) | Knowledge distillation, combined |
| `p7_rescore_saved_runs.ipynb` | | Re-scoring saved runs |
| `p7_rescore_saved_runs_v2.ipynb` | | Re-scoring saved runs, second version |

## Before you run anything

The fine-tuning and distillation notebooks need two things from Phase 1:

1. **The tokenizer output** from the matching BPE notebook.
2. **The pretrained model weights** that came out of the matching Phase 1 model notebook.

Fine-tuning starts from those weights, and the model only makes sense with the exact tokenizer it was pretrained with, so the pairing matters.

| Phase 7 notebook | Tokenizer (from Phase 1) | Pretrained weights (from Phase 1) |
|---|---|---|
| `p7_m1_lorma` | `BPE_model_1_2.ipynb` (35k) | `model1_ed_t.ipynb` |
| `p7_m2p_lora`, `p7_m2p_lorma` | `bpe_final_model2.ipynb` (32k) | `Model2_clean_BPE_final_gelu.ipynb` |
| `p7_m3_lorma+` | `legal_bpe_50k.ipynb` (50k) | `model3_postln_wnorm.ipynb` |
| `p7_KD_3_to_4`, `p7_kd_combined_m3teacher_m4student` | `legal_bpe_50k.ipynb` (50k) | `model3_postln_wnorm.ipynb` and `model4.ipynb` |

The distillation notebooks use two models, so they need the weights for both the teacher and the student.

The two `rescore_saved_runs` notebooks don't train anything new, so they need the saved runs they re-score, along with the tokenizer each of those runs was trained with.

The three tokenizers (35k, 32k and 50k) give different token ids, so they aren't interchangeable. Use the tokenizer that matches the model, otherwise the vocab won't line up and you'll get errors or nonsense output.

## Running a notebook

1. Run the tokenizer notebook from Phase 1 for your model (or reuse its saved output).
2. Get the pretrained weights from the matching Phase 1 model notebook.
3. Point the Phase 7 notebook at both, then run it.
