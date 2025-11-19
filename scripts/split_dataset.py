import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE = "data/original"
DEST = "data"

CLASSES = ["crack", "corrosion", "normal"]

# Create output folders
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(f"{DEST}/{split}/{cls}", exist_ok=True)

def split_class(cls):
    folder = f"{SOURCE}/{cls}"
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".bmp"))]

    train, temp = train_test_split(files, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    return train, val, test

def copy_files(files, src, dst):
    for f in files:
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")

for cls in CLASSES:
    src_folder = f"{SOURCE}/{cls}"

    train_files, val_files, test_files = split_class(cls)

    copy_files(train_files, src_folder, f"{DEST}/train/{cls}")
    copy_files(val_files, src_folder, f"{DEST}/val/{cls}")
    copy_files(test_files, src_folder, f"{DEST}/test/{cls}")

print("Dataset split completed.")
