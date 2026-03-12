import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .predict import get_embedding

DATASET_PATH = "dataset/jobs.csv"


def load_jobs():
    df = pd.read_csv(DATASET_PATH)

    df["text"] = df["Job Title"].fillna("") + " " + df["Job Description"].fillna("")

    return df


def recommend_skills(user_resume, top_k=5):

    df = load_jobs()

    user_vec = get_embedding(user_resume)

    job_vectors = []

    for text in df["text"]:
        vec = get_embedding(text)
        job_vectors.append(vec[0])

    scores = cosine_similarity(user_vec, job_vectors)[0]

    df["score"] = scores

    top_jobs = df.sort_values("score", ascending=False).head(top_k)

    skills = []

    for desc in top_jobs["Job Description"]:
        words = desc.split()

        for word in words:
            if word.lower() in [
                "python", "tensorflow", "pytorch", "docker",
                "kubernetes", "aws", "sql", "machine", "learning"
            ]:
                skills.append(word.lower())

    return list(set(skills))