from model.similarity import similarity

job_desc = "Machine learning engineer with Python TensorFlow"

resume = "Python developer with machine learning and deep learning experience"

score = similarity(job_desc, resume)

print("Match score:", score)