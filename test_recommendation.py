from model.recommend_skills import recommend_skills

resume = """
Python developer with experience in machine learning and data analysis.
"""

skills = recommend_skills(resume)

print("Recommended skills:")
print(skills)