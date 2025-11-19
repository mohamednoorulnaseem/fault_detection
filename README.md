🚀 Project Overview
This project is an AI-powered fault detection system for metal surface defects such as cracks, corrosion, and normal surfaces.
It uses a Convolutional Neural Network (CNN) trained on metal images and provides:

✔ Automated image classification
✔ Interactive Streamlit web app
✔ Training, evaluation, and dataset processing scripts
✔ Model stored using Git LFS

🧠 Model Features
Input size: 224×224

Handles 3 classes:

crack

corrosion

normal

Built using TensorFlow/Keras

Supports .bmp, .png, .jpg images

Includes:

Saved trained model

Confusion matrix

Class index mapping

📂 Project Directory Structure
powershell
Copy code
fault_detection/
│
├── app.py # Streamlit Web App
├── src/
│ ├── train.py # Model training script
│ ├── evaluate.py # Model evaluation script
│ └── preprocess.py # Dataset preprocessing
│
├── scripts/
│ └── split_dataset.py # Train/Validation/Test splitter
│
├── data/
│ ├── original/ # Raw dataset
│ ├── processed/ # Preprocessed images
│ └── sample/ # App test samples
│
├── model.h5 # Model file (LFS)
├── metal_defect_model.h5 # Backup model (LFS)
├── class_indices.json # Class mapping
├── confusion_matrix.png # Eval results
└── README.md
🛠️ Installation
1️⃣ Clone Repository
bash
Copy code
git clone https://github.com/mohamednoorulnaseem/fault_detection.git
cd fault_detection
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv .venv
source .venv/bin/activate # macOS/Linux
.venv\Scripts\activate # Windows
3️⃣ Install Dependencies
nginx
Copy code
pip install -r requirements.txt
4️⃣ Enable Git LFS (required for model files)
nginx
Copy code
git lfs install
git lfs pull
▶️ Running the App
Start the Streamlit Web Application
arduino
Copy code
streamlit run app.py
You will see a local URL such as:

👉 http://localhost:8501
👉 Upload an image → get the defect prediction.

📊 Evaluation
To test the model:

bash
Copy code
python src/evaluate.py
Outputs include:

Accuracy

Confusion matrix

Class predictions

📦 Dataset
Dataset consists of:

300 crack images

300 corrosion images

300 normal images

Stored in:
data/original/<category>/

Format: .bmp

Use script to split dataset:

bash
Copy code
python scripts/split_dataset.py
🧪 Training
To retrain the model:

bash
Copy code
python src/train.py
This will:
✔ Process dataset
✔ Train CNN
✔ Save new model as model.h5

🤝 Contributing
Feel free to open issues or submit pull requests to improve:

Model performance

UI/UX

Dataset quality

Documentation

📬 Contact
Author: Mohamed Noorul Naseem
GitHub: https://github.com/mohamednoorulnaseem
