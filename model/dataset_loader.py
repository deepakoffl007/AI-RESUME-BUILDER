import pandas as pd
from tqdm import tqdm

def load_dataset(path):

    print("Loading dataset...")

    df = pd.read_csv(path)

    texts = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing dataset"):
        title = str(row.get("Job Title", ""))
        desc = str(row.get("Job Description", ""))

        text = title + " " + desc

        texts.append(text)

    print("Total samples:", len(texts))

    return texts