from model.predict import get_embedding

job = "Machine learning engineer with Python TensorFlow deep learning experience"

vector = get_embedding(job)

print("Embedding shape:", len(vector[0]))
print(vector)