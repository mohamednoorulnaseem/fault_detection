import os

def count_images(path):
    total = 0
    for cls in os.listdir(path):
        total += len(os.listdir(os.path.join(path, cls)))
    return total
