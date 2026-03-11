# ResumeAI — Deep Learning Training Pipeline
> Train your own resume generation model locally — no API required.

---

## Architecture Options

| Model | Size | Best For | Speed |
|-------|------|----------|-------|
| `flan-t5-small` | 80M | Quick experiments | Very fast |
| **`flan-t5-base`** | 250M | **Recommended** | Fast |
| `flan-t5-large` | 770M | Best quality | Slow |
| `distilgpt2` | 82M | CPU-only training | Fast |
| `gpt2` | 117M | Balanced | Moderate |

**Recommendation:** Use `flan-t5-base` (seq2seq). It learns the structured-input → resume-output task much better than GPT-2.

---

## Setup

```bash
# 1. Clone / enter project
cd resume_ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (GPU) Install CUDA PyTorch — visit pytorch.org for your CUDA version
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 1 — Prepare Data

```bash
# Using built-in sample data (good for testing):
python data/prepare_dataset.py --model_type t5

# Using your own resume data (recommended for real training):
# Create a file: my_resumes.jsonl
# Each line: {"input": {"name":..., "title":..., "skills":..., "experience":[...]}, "output": "resume text"}
python data/prepare_dataset.py --model_type t5 --custom_data my_resumes.jsonl
```

### Data Format (JSONL)
```json
{
  "input": {
    "name": "Jane Doe",
    "title": "Software Engineer",
    "skills": "Python, React, AWS",
    "experience": [
      {"company": "TechCorp", "role": "Developer", "duration": "2021-2024", "points": "Built APIs serving 1M users"}
    ],
    "education": "BS CS, MIT, 2021"
  },
  "output": "Jane Doe\nSoftware Engineer\n\nPROFESSIONAL SUMMARY\n..."
}
```

### Where to Get Training Data
- **Kaggle Resume Datasets**: search "resume dataset" on kaggle.com
- **GitHub**: search "resume corpus NLP"
- **Synthetic generation**: use ChatGPT/Claude to generate 100+ diverse resume pairs
- **Scraping**: LinkedIn/Indeed (respect ToS)

---

## Step 2 — Train

### Option A: T5 (Recommended)
```bash
# Basic training
python train_t5.py

# With GPU and mixed precision (2-3x faster)
python train_t5.py --fp16 --batch 8

# Larger model for better quality
python train_t5.py --model flan-t5-large --epochs 20

# Resume interrupted training
python train_t5.py --resume_from models/t5_resume/checkpoint-500
```

### Option B: GPT-2
```bash
python train_gpt2.py
python train_gpt2.py --model gpt2-medium --fp16
```

### Google Colab (Free GPU)
```python
# In a Colab cell:
!git clone https://github.com/YOUR_REPO/resume_ai
%cd resume_ai
!pip install -r requirements.txt
!python data/prepare_dataset.py --model_type t5
!python train_t5.py --fp16 --batch 8 --epochs 20
```

---

## Step 3 — Monitor Training

Training logs are printed every 10 steps. Key metrics:
- **`train_loss`** — should decrease each epoch
- **`eval_loss`** — validation loss; if it stops improving, early stopping kicks in
- **`rougeL`** (T5 only) — measures text similarity to ground truth (higher = better)

To use TensorBoard:
```bash
# In train_t5.py, set report_to="tensorboard"
tensorboard --logdir models/t5_resume/logs
```

---

## Step 4 — Run Inference

```bash
# Quick demo
python inference.py --model_path models/t5_resume/final --model_type t5

# Interactive mode (enter resume info manually)
python inference.py --model_type t5 --interactive
```

---

## Project Structure

```
resume_ai/
├── data/
│   ├── prepare_dataset.py    # Data prep + augmentation
│   └── processed/            # Saved HuggingFace datasets
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   ├── t5_resume/            # T5 checkpoints + final model
│   └── gpt2_resume/          # GPT-2 checkpoints + final model
├── train_t5.py               # T5 fine-tuning script
├── train_gpt2.py             # GPT-2 fine-tuning script
├── inference.py              # Run your trained model
├── requirements.txt
└── README.md
```

---

## Estimated Training Times

| Setup | Model | Time |
|-------|-------|------|
| Free Colab T4 GPU | flan-t5-base, 15 epochs | ~30 min |
| RTX 3080 | flan-t5-large, 20 epochs | ~45 min |
| CPU only | distilgpt2, 10 epochs | ~2-3 hours |

---

## Tips for Better Results

1. **More data = better model.** Aim for 500+ diverse resume samples minimum.
2. **Use Flan-T5**, not GPT-2, for structured generation tasks.
3. **Augment your data** — shuffle skill order, vary phrasing.
4. **Use `--fp16`** if you have an NVIDIA GPU — 2x faster training.
5. **Watch eval loss**, not just train loss. Early stopping prevents overfitting.
6. After training, **quantize with ONNX or llama.cpp** for fast CPU inference.
