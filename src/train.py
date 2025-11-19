import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model
import os

DATA_DIR = "data"

BATCH_SIZE = 16
IMG_SIZE = (224, 224)

def get_generators():

    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen = ImageDataGenerator(rescale=1./255)

    train = train_gen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val = val_gen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    return train, val

def main():
    train, val = get_generators()

    model = build_model(num_classes=len(train.class_indices))

    model.fit(
        train,
        validation_data=val,
        epochs=10
    )

    model.save("metal_defect_model.h5")
    print("Training complete. Model saved!")

if __name__ == "__main__":
    main()
