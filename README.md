# Capstone Project: Phase 5

Phase 5 carries on with the fine-tuning. There are LoRA and rsLoRA runs on models 1, 2 (preprocessed) and 3, plus a run that distils model 3 into model 4 and fine-tunes it with LoRA.

## Naming

Most notebooks start with `p5_`, and a few start with `phase5_`. Same phase, just named differently. After the prefix:

- `m1`, `m2p`, `m3` are models 1, 2 (preprocessed) and 3 from Phase 1. Model 1 is the encoder-decoder. `m2p` is model 2 trained on the preprocessed 32k BPE tokenizer, so it's the "clean" version of model 2. The `p` is for preprocessed.
- `KD_3_to_4` is knowledge distillation from model 3 to model 4.
- Extra bits at the end of a name (`allchunks`, `tokpool`, `resum...`) are the variant of that particular run.

## What's in here

| Notebook | Model | Method |
|---|---|---|
| `p5_m1_rslora.ipynb` | Model 1 | rsLoRA |
| `p5_m2p_lora.ipynb` | Model 2 (preprocessed) | LoRA |
| `phase5_lsi_model3_lora_allchunks.ipynb` | Model 3 | LoRA (all chunks) |
| `phase5_lsi_model4_kd_lora_tokpool_resum…` | Model 4 | Knowledge distillation + LoRA (token pooling) |
| `p5_KD_3_to_4.ipynb` | Model 3 to Model 4 | Knowledge distillation |

## Before you run anything

Each notebook needs two things from Phase 1:

1. **The tokenizer output** from the matching BPE notebook.
2. **The pretrained model weights** that came out of the matching Phase 1 model notebook.

Fine-tuning starts from those weights, and the model only makes sense with the exact tokenizer it was pretrained with, so the pairing matters.

| Phase 5 notebook | Tokenizer (from Phase 1) | Pretrained weights (from Phase 1) |
|---|---|---|
| `p5_m1_rslora` | `BPE_model_1_2.ipynb` (35k) | `model1_ed_t.ipynb` |
| `p5_m2p_lora` | `bpe_final_model2.ipynb` (32k) | `Model2_clean_BPE_final_gelu.ipynb` |
| `phase5_lsi_model3_lora_allchunks` | `legal_bpe_50k.ipynb` (50k) | `model3_postln_wnorm.ipynb` |
| `phase5_lsi_model4_kd_lora_tokpool_resum…`, `p5_KD_3_to_4` | `legal_bpe_50k.ipynb` (50k) | `model3_postln_wnorm.ipynb` and `model4.ipynb` |

The distillation notebooks use two models, so they need the weights for both the teacher and the student.

The three tokenizers (35k, 32k and 50k) give different token ids, so they aren't interchangeable. Use the tokenizer that matches the model, otherwise the vocab won't line up and you'll get errors or nonsense output.

## Running a notebook

1. Run the tokenizer notebook from Phase 1 for your model (or reuse its saved output).
2. Get the pretrained weights from the matching Phase 1 model notebook.
3. Point the Phase 5 notebook at both, then run it.
