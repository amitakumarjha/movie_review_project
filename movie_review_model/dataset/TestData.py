import os

file_path = "aclImdb/train/pos/4077_10.txt"

if os.path.exists(file_path):
    with open(file_path, "r") as f:
        print(f.read())
else:
    print(f"File not found: {file_path}")