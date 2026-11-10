# Capstone Project

This repo holds my capstone project: pretraining language models from scratch, with my own BPE tokenizers, and then fine-tuning them in a bunch of different ways.

The work is split into seven phases, and **each phase lives on its own branch**. Phase 1 is the main pretraining phase. Phases 2 to 7 are all fine-tuning. Use the branch dropdown (or the links below) to jump between them. Every branch has its own README that lists the files and explains how to run them.

## The phases

| Branch | What's in it |
|---|---|
| [`phase1`](https://github.com/krishkalgude/capstone-project/tree/phase1) | Pretraining: the BPE tokenizers (35k, 32k, 50k) and models 1 to 4, plus the cleaned-up model 2 and the Longformer |
| [`phase2`](https://github.com/krishkalgude/capstone-project/tree/phase2) | First round of fine-tuning: LoRA, full fine-tuning, prefix fine-tuning, LoRMA, and distillation from model 3 to model 4 |
| [`phase3`](https://github.com/krishkalgude/capstone-project/tree/phase3) | LoRA and its variants (LoRMA, AdaLoRA, rsLoRA), plus another distillation run |
| [`phase4`](https://github.com/krishkalgude/capstone-project/tree/phase4) | Fine-tuning the Longformer, plus more LoRA, rsLoRA and LoRMA runs and distillation |
| [`phase5`](https://github.com/krishkalgude/capstone-project/tree/phase5) | rsLoRA on model 1, LoRA on models 2 (preprocessed) and 3, and distillation with LoRA into model 4 |
| [`phase6`](https://github.com/krishkalgude/capstone-project/tree/phase6) | LoRA, rsLoRA and LoRMA variants and distillation with LoRA, written as Python scripts instead of notebooks |
| [`phase7`](https://github.com/krishkalgude/capstone-project/tree/phase7) | Final round: LoRMA and LoRMA+, combined distillation, and notebooks that re-score saved runs |

If you're new to the repo, start with `phase1`. Everything else builds on it.

## The models

These are all trained in Phase 1 and reused for fine-tuning after that.

| Model | Architecture | Tokenizer |
|---|---|---|
| Model 1 | Encoder-decoder | BPE 35k |
| Model 2 | Decoder-only | BPE 35k |
| Model 2 clean (`m2p`) | Decoder-only | BPE 32k (multilingual, deduplicated, preprocessed) |
| Model 3 | Post-LN version | BPE 50k |
| Model 4 | | BPE 50k |
| Longformer | Longformer | BPE 32k (same as model 2 clean) |

The `m1`, `m2`, `m2p`, `m3`, `m4` names in the Phase 2 to 7 files refer to these. The `p` in `m2p` is for preprocessed.

## What gets tried

Across the fine-tuning phases I tried LoRA, rsLoRA, AdaLoRA, LoRMA (and LoRMA+), prefix fine-tuning, full fine-tuning, and knowledge distillation from model 3 into model 4.

## How the pieces fit together

The one rule to remember: **a fine-tuning notebook always needs the tokenizer and the pretrained weights from Phase 1 for the model it's fine-tuning.**

- Each model only works with the tokenizer it was pretrained with. The 35k, 32k and 50k tokenizers give different token ids, so they can't be swapped.
- Fine-tuning starts from the Phase 1 weights, so you need the output of the matching Phase 1 model notebook too.
- Distillation runs use two models (model 3 as the teacher, model 4 as the student), so they need the weights for both.

The README on each branch has a table showing exactly which Phase 1 notebooks each file depends on.
