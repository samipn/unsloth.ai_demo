# Unsloth.ai Demos — Colab-Friendly Fine-Tuning & RL

A compact set of **Google Colab notebooks** for fine-tuning and aligning small-to-mid LLMs with [Unsloth](https://github.com/unslothai/unsloth).  
These notebooks cover **full fine-tuning, LoRA/QLoRA, preference learning (DPO), GRPO for reasoning on GSM8K**, and **continued pretraining**.

Walkthrough video link: https://drive.google.com/file/d/1ITNVUS7i1a7fm_bxW0-3hPsc5rn0HDa1/view?usp=sharing

> 📝 Tip: If GitHub ever fails to render a notebook, use the **Open in Colab** buttons below — they always work.

---

## 📒 Notebooks

| Notebook | What it does | Open in Colab |
|---|---|---|
| `colab1_full_finetune_smollm2.ipynb` | **Full fine-tune** of a compact model (e.g., SmolLM2) on your dataset. Good baseline to understand Unsloth training loops end‑to‑end. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samipn/unsloth.ai_demo/blob/main/colab1_full_finetune_smollm2.ipynb) |
| `colab2_lora_smollm2.ipynb` | **LoRA/QLoRA** fine-tuning for low VRAM settings while retaining strong performance. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samipn/unsloth.ai_demo/blob/main/colab2_lora_smollm2.ipynb) |
| `colab3_rl_dpo_gemma1b.ipynb` | **Preference learning with DPO** on a 1B‑class Gemma model (and RL‑ready setup). Use this to align responses to human preferences without training a reward model. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samipn/unsloth.ai_demo/blob/main/colab3_rl_dpo_gemma1b.ipynb) |
| `colab4_grpo_gsm8k_gemma1b.ipynb` | **GRPO reasoning RL** on **GSM8K** for Gemma‑1B: turn a base model into a better step‑by‑step reasoner. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samipn/unsloth.ai_demo/blob/main/colab4_grpo_gsm8k_gemma1b.ipynb) |
| `colab5_continued_pretraining.ipynb` | **Continued pretraining / domain‑adaptive pretraining** on your own corpus before SFT/LoRA/GRPO. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samipn/unsloth.ai_demo/blob/main/colab5_continued_pretraining.ipynb) |

---

## 🚀 Quickstart (Colab)

1. Click a **Colab** button above.
2. **Runtime → Change runtime type → GPU** (A100 recommended).
3. Run the install/setup cells — the notebooks install **Unsloth** and dependencies.
4. For **gated models** (e.g., some Gemma checkpoints), add your Hugging Face token:
   ```python
   import os
   os.environ["HUGGING_FACE_HUB_TOKEN"] = "hf_xxx"  # replace with your token
   ```
5. Point the notebook to your **dataset** (local upload, Google Drive, or Hugging Face `datasets` hub).
6. Hit **Run all** and follow the in‑notebook prompts.

---

## 🧰 What’s inside / typical flows

- **Full fine-tune (SmolLM2)** — Start with a small, efficient model to validate your pipeline end‑to‑end before scaling.
- **LoRA/QLoRA** — Adapter‑based training to cut VRAM while staying performant.
- **DPO alignment (Gemma‑1B)** — Align generations to human preferences *without* training a reward model.
- **GRPO on GSM8K (Gemma‑1B)** — Reinforcement learning that improves step‑by‑step mathematical reasoning.
- **Continued pretraining** — Warm up a base model on in‑domain text before instruction fine‑tuning.

---

## 🗂️ Repository structure

```
unsloth.ai_demo/
├─ colab1_full_finetune_smollm2.ipynb
├─ colab2_lora_smollm2.ipynb
├─ colab3_rl_dpo_gemma1b.ipynb
├─ colab4_grpo_gsm8k_gemma1b.ipynb
├─ colab5_continued_pretraining.ipynb
└─ README.md  ← you are here
```

---

## ⚠️ Notes & good practices

- **Tokens & TOS**: Accept the model’s license (e.g., Gemma) on Hugging Face and use your token in Colab.
- **VRAM**: LoRA/QLoRA notebooks are friendlier for T4‑class GPUs; full fine‑tuning may require more VRAM.
- **Reproducibility**: Set `seed` where available; log runs and checkpoints to Drive or W&B as desired.
- **Datasets**: For GRPO math experiments, try [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k).

---
