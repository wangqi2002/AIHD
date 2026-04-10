import os

# dir_path = os.path.dirname(__file__)
# print(dir_path)

with open("./scene.txt","r")as f:
    prompt = f.read()
print(prompt)