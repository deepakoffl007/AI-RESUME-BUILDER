import pandas as pd
import numpy as np
from tqdm import tqdm
from .predict import get_embedding

DATASET_PATH = "dataset/jobs.csv"
OUTPUT_PATH = "saved_model/job_embeddings.npy"


def build_embeddings():

    df = pd.read_csv(DATASET_PATH)

    texts = df["Job Title"].fillna("") + " " + df["Job Description"].fillna("")

    embeddings = []

    print("Building job embeddings...")

    for text in tqdm(texts):
        vec = get_embedding(text)
        embeddings.append(vec[0])

    embeddings = np.array(embeddings)

    np.save(OUTPUT_PATH, embeddings)

    print("Saved embeddings to:", OUTPUT_PATH)


if __name__ == "__main__":
    build_embeddings()