from sklearn.metrics.pairwise import cosine_similarity
from .predict import get_embedding

def similarity(text1, text2):

    vec1 = get_embedding(text1)
    vec2 = get_embedding(text2)

    score = cosine_similarity(vec1, vec2)

    return score[0][0]