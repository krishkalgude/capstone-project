# Capstone Project: Phase 1

This branch holds everything from Phase 1, the main pretraining phase of the capstone. There are two parts to it: building the BPE tokenizers, and pretraining the models on top of them. The fine-tuning phases come later and build on what's trained here.

All the code is in Jupyter notebooks. Nothing fancy, just one notebook per tokenizer and one per model.

## What's in here

| Notebook | What it is |
|---|---|
| `BPE_model_1_2.ipynb` | BPE tokenizer, 35k vocab (for model 1 and model 2) |
| `bpe_final_model2.ipynb` | BPE tokenizer, 32k vocab, the cleaned-up one |
| `legal_bpe_50k.ipynb` | BPE tokenizer, 50k vocab (for model 3 and model 4) |
| `model1_ed_t.ipynb` | Model 1, encoder-decoder |
| `model2_decoder_only_final.ipynb` | Model 2, decoder-only |
| `Model2_clean_BPE_final_gelu.ipynb` | Model 2 retrained on the cleaned 32k tokenizer (GELU) |
| `model3_postln_wnorm.ipynb` | Model 3, post-LN version |
| `model4.ipynb` | Model 4 |
| `longformer-f.ipynb` | Longformer, trained on the cleaned 32k tokenizer |

## The tokenizers

I ended up with three BPE tokenizers over the course of this phase.

**35k** was the first one. Models 1 and 2 were both trained with it.

**32k** is the cleaned-up rebuild. This one handles multilingual text properly, the data was deduplicated before training it, and the text goes through preprocessing first. Model 2 (clean) and the Longformer both use this tokenizer, so they're the ones to look at if you want to see the effect of cleaner data.

**50k** is the bigger vocabulary, used for models 3 and 4.

## The models

- **Model 1** is an encoder-decoder, trained on the 35k tokenizer.
- **Model 2** is decoder-only, also on the 35k tokenizer.
- **Model 2 (clean)** is the same idea as model 2, but retrained on the 32k tokenizer with the deduplicated, multilingual-aware data.
- **Model 3** and **Model 4** are trained on the 50k tokenizer. Model 3 is the post-LN one.
- **Longformer** is trained on the same preprocessed 32k tokenizer as Model 2 (clean).

## Running a model

One thing to know before you run anything: **every model needs the output of its own tokenizer notebook as input.** The model's embedding size and token ids are tied to the vocab it was trained with, so a model won't work with a different tokenizer, even one that looks similar.

So the order is always:

1. Run the tokenizer notebook for your model (or reuse its saved output if you already have it).
2. Point the model notebook at that tokenizer output.
3. Run the model notebook.

Which tokenizer goes with which model:

| Model | Tokenizer notebook |
|---|---|
| `model1_ed_t.ipynb` | `BPE_model_1_2.ipynb` (35k) |
| `model2_decoder_only_final.ipynb` | `BPE_model_1_2.ipynb` (35k) |
| `Model2_clean_BPE_final_gelu.ipynb` | `bpe_final_model2.ipynb` (32k) |
| `longformer-f.ipynb` | `bpe_final_model2.ipynb` (32k) |
| `model3_postln_wnorm.ipynb` | `legal_bpe_50k.ipynb` (50k) |
| `model4.ipynb` | `legal_bpe_50k.ipynb` (50k) |

Mixing them up (say, model 3 with the 35k tokenizer) will either error out or give you garbage, so double-check the pairing first.
