"""
train_gpt2.py
-------------
Fine-tunes GPT-2 (or DistilGPT2) on resume generation using causal LM.
GPT-2 learns to continue: [RESUME REQUEST] → [FULL RESUME]

WHY GPT-2?
  ✓ Lightweight, trainable on CPU or small GPU
  ✓ Great for open-ended text generation
  ✓ Checkpoint available on HuggingFace, no API needed

USAGE:
  python train_gpt2.py
  python train_gpt2.py --model gpt2-medium --epochs 10
"""

import os
import argparse
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)

MODELS = {
    "distilgpt2": "distilgpt2",    # ~82M, fastest
    "gpt2":       "gpt2",          # ~117M, recommended
    "gpt2-medium":"gpt2-medium",   # ~345M, better quality
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="gpt2", choices=MODELS.keys())
    p.add_argument("--data_dir",    default="data/processed")
    p.add_argument("--output_dir",  default="models/gpt2_resume")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch",       type=int,   default=4)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--max_length",  type=int,   default=768)
    p.add_argument("--fp16",        action="store_true")
    return p.parse_args()


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = MODELS[args.model]

    print(f"\n{'='*50}")
    print(f"  Model:  {model_name}  |  Device: {device}")
    print(f"{'='*50}\n")

    # ── Load tokenizer & model ──
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token by default

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # ── Load datasets ──
    train_ds = load_from_disk(f"{args.data_dir}/train")
    val_ds   = load_from_disk(f"{args.data_dir}/val")

    # ── Tokenize ──
    tok_fn = lambda x: tokenize_function(x, tokenizer, args.max_length)
    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_tok   = val_ds.map(tok_fn, batched=True, remove_columns=["text"])

    # ── Data collator (causal LM: no masking, just predict next token) ──
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked LM
    )

    # ── Training args ──
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,

        logging_steps=10,
        report_to="none",
        fp16=(args.fp16 and device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("🚀 Starting GPT-2 fine-tuning...\n")
    trainer.train()

    final_path = f"{args.output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ GPT-2 model saved to: {final_path}")

    # ── Quick test ──
    print("\n🧪 Quick generation test...")
    model.eval()
    prompt = (
        "### RESUME REQUEST\n"
        "Name: Jordan Lee\n"
        "Title: UX Designer\n"
        "Skills: Figma, Sketch, User Research, Prototyping, CSS\n"
        "Experience: UX Designer at DesignStudio (2020-2024): redesigned onboarding flow, +40% conversion\n"
        "Education: BFA Design, RISD, 2020\n"
        "### RESUME\n"
    )
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=400,
            temperature=0.7,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(output[0], skip_special_tokens=True)
    resume_part = full.split("### RESUME\n")[-1].split("<|endoftext|>")[0].strip()
    print("\n" + "─"*50)
    print(resume_part[:600])
    print("─"*50)


if __name__ == "__main__":
    main()
