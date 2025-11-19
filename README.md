# 🛠️ Metal Surface Fault Detection

### Crack • Corrosion • Normal Surface Classification Using Deep Learning

![Banner](https://raw.githubusercontent.com/mohamednoorulnaseem/fault_detection/main/assets/banner.png)

---

## 🚀 Overview

This project is a **deep learning–based metal surface defect detection system** that identifies:

- **Crack**
- **Corrosion**
- **Normal surface**

It uses a trained **CNN model (.h5)** and a clean **Streamlit web app** for image-based predictions.  
Perfect for industrial inspection, automation, and quality control.

---

## 📂 Project Structure

```
fault_detection/
│── app.py                      # Streamlit web application
│── metal_defect_model.h5       # Trained defect detection model (LFS)
│── class_indices.json          # Class label mappings
│── data/
│   ├── original/
│   │   ├── crack/              # Crack images
│   │   ├── corrosion/          # Corrosion images
│   │   ├── normal/             # Normal metal surface images
│   └── sample/                 # Sample test images
│── scripts/
│   ├── train.py                # Model training script
│   ├── split_dataset.py        # Train-test split script
│── README.md                   # Project documentation
│── requirements.txt            # Dependencies
```

---

## 🧠 Features

✔ Classifies **3 types of metal surface conditions**  
✔ Uses a **TensorFlow / Keras CNN model**  
✔ Includes **dataset split script**  
✔ Comes with a ready-to-run **Streamlit UI**  
✔ Git LFS enabled for large model files

---

## 🖥️ Run Locally

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

### 3️⃣ Upload any metal surface image

The model will output:

- Prediction label
- Confidence score

---

## 🧪 Training the Model (Optional)

To retrain the model using your own dataset:

```bash
python scripts/train.py
```

To regenerate dataset splits:

```bash
python scripts/split_dataset.py
```

---

## 📦 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **Streamlit**
- **Git LFS**

---

## 🎯 Future Improvements

- Add real-time video defect detection
- Improve model accuracy with augmentation
- Deploy on cloud (AWS / GCP / Azure)

---

## 🙌 Author

**Mohamed Noorul Naseem**  
GitHub: [mohamednoorulnaseem](https://github.com/mohamednoorulnaseem)
