# 🌿 PlantGuard

**PlantGuard** is an intelligent image classification system built to detect and classify diseases in the leaves of perennial shrubs, focusing primarily on crops like **coffee** and **tea**. Powered by deep learning and an EfficientNet-based Convolutional Neural Network (CNN), PlantGuard delivers high-accuracy predictions to support early disease diagnosis and precision agriculture.

---

## 🚀 Features

- ✅ Image classification using CNN (EfficientNet)
- ✅ Handles multiple disease categories
- ✅ Flask-based web interface for easy uploads and results
- ✅ Model training history tracking
- ✅ Scalable to add new plant species or diseases

---

## 🧠 Model Architecture

- Backbone: **EfficientNetB0**
- Input Shape: 224x224x3
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy
- Includes data augmentation, early stopping, and checkpointing

---


## Link to download the dataset
https://drive.google.com/drive/folders/1t078VMKcXacKM0kvS2Z7-BHq5oxd9K3u?usp=sharing


- Each class folder should contain images of a specific disease or healthy leaf.
- Resize all images to **224x224**.

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/plantguard.git
   cd plantguard
   
2. Create a Virtual Environment

   ```bash
   python -m venv env
   # Activate the environment
   # On macOS/Linux:
   source env/bin/activate
   # On Windows:
   env\Scripts\activate

3. Install Dependencies
   ```bash
   pip install -r requirements.txt

4. Train the Model

5. Run the Flask App
   ```bash
   python app.py
