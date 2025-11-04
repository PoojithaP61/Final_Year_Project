import os
for root, dirs, files in os.walk("data"):
    if files:
        print(root, len(files))