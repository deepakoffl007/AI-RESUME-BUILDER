"""
train_t5.py
-----------
Fine-tunes google/flan-t5-base (or t5-small) on resume generation.
T5 is a seq2seq model: structured input → full resume text.

WHY T5?
  ✓ Smaller than LLaMA, trains on free Colab GPU
  ✓ seq2seq is ideal for structured-input → text-output tasks
  ✓ Flan-T5 already instruction-tuned, learns resume format fast

USAGE:
  python train_t5.py                          # default settings
  python train_t5.py --model flan-t5-large    # bigger model
  python train_t5.py --epochs 10 --batch 4    # custom hyperparams
"""

import os
import argparse
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate
import numpy as np

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODELS = {
    "flan-t5-small":  "google/flan-t5-small",   # ~80M params, fastest
    "flan-t5-base":   "google/flan-t5-base",    # ~250M params, recommended
    "flan-t5-large":  "google/flan-t5-large",   # ~770M params, best quality
    "t5-small":       "t5-small",
    "t5-base":        "t5-base",
}

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune T5 for resume generation")
    p.add_argument("--model",        default="flan-t5-base", choices=MODELS.keys())
    p.add_argument("--data_dir",     default="data/processed")
    p.add_argument("--output_dir",   default="models/t5_resume")
    p.add_argument("--epochs",       type=int,   default=15)
    p.add_argument("--batch",        type=int,   default=4)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--max_input",    type=int,   default=512)
    p.add_argument("--max_target",   type=int,   default=512)
    p.add_argument("--warmup_steps", type=int,   default=100)
    p.add_argument("--fp16",         action="store_true", help="Use mixed precision (requires GPU)")
    p.add_argument("--resume_from",  default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        print(f"🖥  GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("🖥  Apple Silicon MPS")
        return "mps"
    else:
        print("⚠️  No GPU found — training on CPU (will be slow)")
        return "cpu"


def preprocess_function(examples, tokenizer, max_input, max_target):
    """Tokenize source/target pairs."""
    model_inputs = tokenizer(
        examples["source"],
        max_length=max_input,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        examples["target"],
        max_length=max_target,
        truncation=True,
        padding="max_length",
    )
    # Replace pad token IDs in labels with -100 so they're ignored in loss
    label_ids = labels["input_ids"]
    label_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in lbl]
        for lbl in label_ids
    ]
    model_inputs["labels"] = label_ids
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """Compute ROUGE scores for evaluation."""
    rouge = evaluate.load("rouge")
    preds, labels = eval_preds

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: round(v * 100, 2) for k, v in result.items()}


def main():
    args = parse_args()
    device = get_device()

    model_name = MODELS[args.model]
    print(f"\n{'='*50}")
    print(f"  Model:      {model_name}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Learn rate: {args.lr}")
    print(f"{'='*50}\n")

    # ── Load tokenizer & model ──
    print("📥 Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Parameters: {total_params:.1f}M")

    # ── Load datasets ──
    print("📂 Loading datasets...")
    train_ds = load_from_disk(f"{args.data_dir}/train")
    val_ds   = load_from_disk(f"{args.data_dir}/val")
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Tokenize ──
    print("🔤 Tokenizing...")
    tokenize_fn = lambda x: preprocess_function(
        x, tokenizer, args.max_input, args.max_target
    )
    train_tokenized = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    val_tokenized   = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)

    # ── Training arguments ──
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        # Evaluation & saving
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        save_total_limit=3,

        # Generation
        predict_with_generate=True,
        generation_max_length=args.max_target,

        # Logging
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        report_to="none",  # change to "tensorboard" if you want

        # Performance
        fp16=(args.fp16 and device == "cuda"),
        dataloader_num_workers=2,

        # Resume training
        resume_from_checkpoint=args.resume_from,
    )

    # ── Data collator ──
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    # ── Trainer ──
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ──
    print("\n🚀 Starting training...\n")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # ── Save final model ──
    final_path = f"{args.output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Model saved to: {final_path}")

    # ── Quick test ──
    print("\n🧪 Quick inference test...")
    model.eval()
    test_input = (
        "generate resume: "
        "name: Jane Doe "
        "title: Machine Learning Engineer "
        "skills: Python, PyTorch, TensorFlow, Docker, Kubernetes "
        "experience: ML Engineer at AI Startup (2021-2024): Built NLP pipelines serving 500K users "
        "education: MS Machine Learning, Carnegie Mellon, 2021"
    )
    inputs = tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "─"*50)
    print(result[:800])
    print("─"*50)

    return trainer


if __name__ == "__main__":
    main()
