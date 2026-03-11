"""
inference.py
------------
Run inference with your trained resume generation model.
Works with both T5 and GPT-2 trained models.

USAGE:
  python inference.py --model_path models/t5_resume/final --model_type t5
  python inference.py --model_path models/gpt2_resume/final --model_type gpt2
  python inference.py --interactive   # chat-style input
"""

import argparse
import torch
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2TokenizerFast,
)


def load_model(model_path: str, model_type: str):
    print(f"📥 Loading {model_type.upper()} model from {model_path}...")
    if model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Model ready on {device}\n")
    return model, tokenizer, device


def generate_t5(model, tokenizer, device, user_data: dict,
                num_beams=4, max_length=512) -> str:
    exp_text = " | ".join(
        f"{e['role']} at {e['company']} ({e['duration']}): {e['points']}"
        for e in user_data.get("experience", [])
    )
    source = (
        f"generate resume: "
        f"name: {user_data['name']} "
        f"title: {user_data.get('title', '')} "
        f"skills: {user_data.get('skills', '')} "
        f"experience: {exp_text} "
        f"education: {user_data.get('education', '')}"
    )
    inputs = tokenizer(source, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.5,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_gpt2(model, tokenizer, device, user_data: dict,
                  max_new_tokens=400, temperature=0.7) -> str:
    exp_text = "\n".join(
        f"{e['role']} at {e['company']} ({e['duration']}): {e['points']}"
        for e in user_data.get("experience", [])
    )
    prompt = (
        f"### RESUME REQUEST\n"
        f"Name: {user_data['name']}\n"
        f"Title: {user_data.get('title', '')}\n"
        f"Skills: {user_data.get('skills', '')}\n"
        f"Experience: {exp_text}\n"
        f"Education: {user_data.get('education', '')}\n"
        f"### RESUME\n"
    )
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return full_text.split("### RESUME\n")[-1].split("<|endoftext|>")[0].strip()


def interactive_mode(model, tokenizer, device, model_type):
    print("=" * 50)
    print("  ResumeAI — Interactive Mode")
    print("  Type 'quit' to exit")
    print("=" * 50 + "\n")

    while True:
        print("\n📝 Enter resume details:")
        name  = input("  Full Name:         ").strip()
        if name.lower() == "quit":
            break
        title  = input("  Target Job Title:  ").strip()
        skills = input("  Skills (comma sep):{} ").format("").strip()
        company= input("  Company:           ").strip()
        role   = input("  Your Role:         ").strip()
        duration = input("  Duration:          ").strip()
        points = input("  Key achievements:  ").strip()
        edu    = input("  Education:         ").strip()

        user_data = {
            "name": name, "title": title, "skills": skills,
            "education": edu,
            "experience": [{"company": company, "role": role,
                             "duration": duration, "points": points}]
        }

        print("\n⚡ Generating resume...\n")
        if model_type == "t5":
            result = generate_t5(model, tokenizer, device, user_data)
        else:
            result = generate_gpt2(model, tokenizer, device, user_data)

        print("─" * 50)
        print(result)
        print("─" * 50)

        save = input("\nSave to file? (y/n): ").strip().lower()
        if save == "y":
            fname = f"resume_{name.replace(' ', '_').lower()}.txt"
            with open(fname, "w") as f:
                f.write(result)
            print(f"💾 Saved to {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/t5_resume/final")
    parser.add_argument("--model_type", default="t5", choices=["t5", "gpt2"])
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_path, args.model_type)

    if args.interactive:
        interactive_mode(model, tokenizer, device, args.model_type)
    else:
        # Demo run
        demo_data = {
            "name": "Sam Rivera",
            "title": "Full Stack Developer",
            "skills": "TypeScript, React, Node.js, GraphQL, PostgreSQL, Redis",
            "education": "BS Computer Engineering, Georgia Tech, 2022",
            "experience": [{
                "company": "WebAgency",
                "role": "Full Stack Developer",
                "duration": "2022-2024",
                "points": "Built SaaS platform with 10K users, reduced load time by 60%"
            }]
        }
        print("⚡ Generating demo resume...\n")
        if args.model_type == "t5":
            result = generate_t5(model, tokenizer, device, demo_data)
        else:
            result = generate_gpt2(model, tokenizer, device, demo_data)
        print("─" * 50)
        print(result)
        print("─" * 50)


if __name__ == "__main__":
    main()
