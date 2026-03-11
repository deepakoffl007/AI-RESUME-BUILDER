"""
prepare_dataset.py
------------------
Creates and preprocesses training data for the resume generation model.
Supports two formats:
  1. GPT-2 style: raw text completion
  2. T5/seq2seq style: structured input -> resume output pairs
"""

import json
import os
import random
from pathlib import Path
from datasets import Dataset
import pandas as pd

# ──────────────────────────────────────────────
# SAMPLE TRAINING DATA  (replace/extend with real resumes)
# ──────────────────────────────────────────────
SAMPLE_DATA = [
    {
        "input": {
            "name": "Alex Johnson",
            "title": "Software Engineer",
            "email": "alex@email.com",
            "skills": "Python, React, AWS, Docker, PostgreSQL",
            "experience": [
                {"company": "TechCorp", "role": "Senior Developer", "duration": "2020-2024",
                 "points": "Built microservices handling 1M requests/day, led team of 4 engineers"}
            ],
            "education": "BS Computer Science, MIT, 2020"
        },
        "output": """Alex Johnson
Software Engineer | alex@email.com

PROFESSIONAL SUMMARY
Results-driven Software Engineer with 4+ years of experience designing and deploying scalable systems. Proven track record building high-traffic microservices and leading cross-functional teams to deliver production-ready solutions.

TECHNICAL SKILLS
Languages & Frameworks: Python, React, Docker
Cloud & Databases: AWS, PostgreSQL
Competencies: Microservices Architecture, Team Leadership, CI/CD

WORK EXPERIENCE
Senior Developer — TechCorp (2020–2024)
• Architected and deployed microservices infrastructure processing 1M+ daily requests, improving system reliability by 40%
• Led and mentored a team of 4 engineers, reducing sprint delivery time by 25%
• Implemented automated testing pipelines that decreased production bugs by 60%

EDUCATION
BS Computer Science — MIT (2020)"""
    },
    {
        "input": {
            "name": "Priya Sharma",
            "title": "Data Scientist",
            "email": "priya@email.com",
            "skills": "Python, TensorFlow, Scikit-learn, SQL, Tableau, R",
            "experience": [
                {"company": "DataCo", "role": "Data Scientist", "duration": "2021-2024",
                 "points": "Built ML models that increased revenue by $2M, created dashboards for C-suite"}
            ],
            "education": "MS Data Science, Stanford, 2021"
        },
        "output": """Priya Sharma
Data Scientist | priya@email.com

PROFESSIONAL SUMMARY
Innovative Data Scientist with 3+ years transforming complex datasets into actionable business insights. Specializes in building end-to-end machine learning pipelines and data visualizations that drive measurable revenue growth.

TECHNICAL SKILLS
ML & AI: TensorFlow, Scikit-learn, R
Data & Analytics: Python, SQL, Tableau
Competencies: Predictive Modeling, Statistical Analysis, Executive Reporting

WORK EXPERIENCE
Data Scientist — DataCo (2021–2024)
• Developed predictive ML models that directly contributed to $2M+ annual revenue increase
• Designed and maintained executive dashboards used by C-suite to guide quarterly strategy
• Reduced data processing time by 50% through optimized ETL pipeline redesign

EDUCATION
MS Data Science — Stanford University (2021)"""
    },
    {
        "input": {
            "name": "Marcus Williams",
            "title": "Product Manager",
            "email": "marcus@email.com",
            "skills": "Product Strategy, Agile, Jira, SQL, User Research, Roadmapping",
            "experience": [
                {"company": "StartupXYZ", "role": "Product Manager", "duration": "2019-2024",
                 "points": "Launched 3 products with 50K+ users, grew retention by 35%"}
            ],
            "education": "MBA, Wharton, 2019"
        },
        "output": """Marcus Williams
Product Manager | marcus@email.com

PROFESSIONAL SUMMARY
Strategic Product Manager with 5+ years launching consumer and B2B products from 0 to 50K+ users. Expert at bridging technical teams and business stakeholders to ship products that measurably improve user retention and revenue.

CORE COMPETENCIES
Product & Strategy: Product Roadmapping, Go-to-Market Strategy, Agile/Scrum
Tools: Jira, SQL, Figma
Research: User Interviews, A/B Testing, Data-Driven Decision Making

WORK EXPERIENCE
Product Manager — StartupXYZ (2019–2024)
• Conceptualized and launched 3 products from idea to 50,000+ active users within 18 months
• Drove a 35% improvement in 90-day user retention through targeted onboarding redesign
• Coordinated cross-functional teams of 12 across engineering, design, and marketing

EDUCATION
MBA — The Wharton School, University of Pennsylvania (2019)"""
    },
]

def format_input_for_gpt2(sample: dict) -> str:
    """Format a sample as a single text block for GPT-2 causal LM training."""
    inp = sample["input"]
    exp_text = "\n".join(
        f"{e['role']} at {e['company']} ({e['duration']}): {e['points']}"
        for e in inp.get("experience", [])
    )
    prompt = (
        f"### RESUME REQUEST\n"
        f"Name: {inp['name']}\n"
        f"Title: {inp['title']}\n"
        f"Skills: {inp['skills']}\n"
        f"Experience: {exp_text}\n"
        f"Education: {inp['education']}\n"
        f"### RESUME\n"
        f"{sample['output']}\n"
        f"<|endoftext|>"
    )
    return prompt


def format_input_for_t5(sample: dict) -> dict:
    """Format a sample as source/target pair for T5 seq2seq training."""
    inp = sample["input"]
    exp_text = " | ".join(
        f"{e['role']} at {e['company']} ({e['duration']}): {e['points']}"
        for e in inp.get("experience", [])
    )
    source = (
        f"generate resume: "
        f"name: {inp['name']} "
        f"title: {inp['title']} "
        f"skills: {inp['skills']} "
        f"experience: {exp_text} "
        f"education: {inp['education']}"
    )
    return {"source": source, "target": sample["output"]}


def augment_data(samples: list, n_augments: int = 3) -> list:
    """
    Simple augmentation: shuffle skill order and paraphrase job title slightly.
    In production, use back-translation or LLM paraphrasing.
    """
    augmented = list(samples)
    for _ in range(n_augments):
        for s in samples:
            new = json.loads(json.dumps(s))  # deep copy
            skills = new["input"]["skills"].split(", ")
            random.shuffle(skills)
            new["input"]["skills"] = ", ".join(skills)
            augmented.append(new)
    return augmented


def build_datasets(output_dir: str = "data/processed", model_type: str = "t5"):
    """
    Build train/validation/test splits and save as HuggingFace datasets.
    
    Args:
        output_dir: where to save processed datasets
        model_type: 't5' for seq2seq or 'gpt2' for causal LM
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Augment data
    all_samples = augment_data(SAMPLE_DATA, n_augments=5)
    random.shuffle(all_samples)

    # Split 80/10/10
    n = len(all_samples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    train_raw = all_samples[:train_end]
    val_raw   = all_samples[train_end:val_end]
    test_raw  = all_samples[val_end:]

    if model_type == "gpt2":
        train_texts = [format_input_for_gpt2(s) for s in train_raw]
        val_texts   = [format_input_for_gpt2(s) for s in val_raw]
        test_texts  = [format_input_for_gpt2(s) for s in test_raw]

        train_ds = Dataset.from_dict({"text": train_texts})
        val_ds   = Dataset.from_dict({"text": val_texts})
        test_ds  = Dataset.from_dict({"text": test_texts})

    else:  # t5
        def to_dict_list(samples):
            return [format_input_for_t5(s) for s in samples]

        train_ds = Dataset.from_list(to_dict_list(train_raw))
        val_ds   = Dataset.from_list(to_dict_list(val_raw))
        test_ds  = Dataset.from_list(to_dict_list(test_raw))

    # Save
    train_ds.save_to_disk(f"{output_dir}/train")
    val_ds.save_to_disk(f"{output_dir}/val")
    test_ds.save_to_disk(f"{output_dir}/test")

    print(f"✅ Dataset built ({model_type})")
    print(f"   Train: {len(train_ds)} samples")
    print(f"   Val:   {len(val_ds)} samples")
    print(f"   Test:  {len(test_ds)} samples")
    print(f"   Saved to: {output_dir}/")
    return train_ds, val_ds, test_ds


def load_custom_jsonl(path: str) -> list:
    """
    Load your own resume data from a JSONL file.
    Each line should be: {"input": {...}, "output": "resume text"}
    """
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"Loaded {len(samples)} samples from {path}")
    return samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="t5", choices=["t5", "gpt2"])
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--custom_data", default=None, help="Path to custom .jsonl file")
    args = parser.parse_args()

    build_datasets(args.output_dir, args.model_type)
