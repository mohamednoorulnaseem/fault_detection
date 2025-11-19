# src/evaluate.py
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "model.h5"   # change if your file name differs

# load mapping
with open("class_indices.json","r") as f:
    class_map = json.load(f)
inv_map = {v:k for k,v in class_map.items()}
class_names = [inv_map[i] for i in range(len(inv_map))]

# load model
model = load_model(MODEL_PATH)

# prepare test generator
gen = ImageDataGenerator(rescale=1./255)
test = gen.flow_from_directory("data/test", target_size=(224,224), batch_size=16, class_mode="categorical", shuffle=False)

# predict
y_true = test.classes
y_prob = model.predict(test, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

# report
print(classification_report(y_true, y_pred, target_names=class_names))
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

# plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")
plt.show()
