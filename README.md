# Chest-Disease-Classification-using-CT-scan
## Workflows
Update config.yaml
Update params.yaml
Update the entity
Update the configuration manager in src config
Update the components
Update the pipeline
Update the main.py
Update the dvc.yaml













# 🏥 Chest Disease Classification using CT Scans  

## 📌 Overview  

This project is an **AI-powered medical diagnosis system** that classifies chest diseases from **CT scan images** using **deep learning and transfer learning**. It provides a scalable and automated solution for early disease detection, helping radiologists and medical professionals make **accurate diagnoses**.  

---

## 🎯 Objective  

🔹 Develop a **high-accuracy** multi-class classification model for chest diseases.  
🔹 Utilize **transfer learning** for enhanced feature extraction.  
🔹 Enable **automated tracking of experiments** using MLflow.  
🔹 Implement **data versioning** with DVC for reproducibility.  
🔹 Deploy the model using **FastAPI on AWS EC2** with Docker and CI/CD.  

---

## 📌 Key Features  

✅ **Multi-class Classification** – Detects multiple chest diseases from CT scans.  
✅ **96% Accuracy** – Achieved high performance using transfer learning.  
✅ **MLflow for Experiment Tracking** – Compares different model versions.  
✅ **DVC for Data Versioning** – Ensures dataset consistency.  
✅ **FastAPI for Deployment** – Serves predictions via REST API.  
✅ **AWS EC2 + Docker** – Scalable cloud-based deployment with CI/CD.  

---

## 🔍 How It Works?  

### 📌 Step 1: Data Preprocessing  
- CT scan images are **preprocessed and normalized** for uniformity.  
- **Data augmentation** is applied to improve generalization.  

### 📌 Step 2: Model Training  
- Uses **pretrained CNN models** (e.g., ResNet]) for feature extraction.  
- Trains a **multi-class classifier** on labeled CT scan images.  
- Optimized with **Adam optimizer** and **categorical cross-entropy loss**.  

### 📌 Step 3: Experiment Tracking with MLflow  
- Logs **hyperparameters, metrics, and model versions**.  
- Compares multiple experiments to select the **best-performing model**.  

### 📌 Step 4: Deployment with FastAPI & AWS  
- The trained model is deployed as a **REST API** using FastAPI.  
- Docker containerization ensures **portability**.  
- **CI/CD pipeline** automates deployment on **AWS EC2**.  

### 📌 Step 5: Prediction & Disease Detection  
- Users upload **CT scan images** via API.  
- The model returns the **predicted disease category**.  

---

## 🏢 Use Cases  

🔹 **Medical Diagnosis** – Assists radiologists in detecting chest diseases early.  
🔹 **Telemedicine** – Enables remote diagnosis through AI-powered scans.  
🔹 **Clinical Research** – Provides AI-driven insights into disease patterns.  
🔹 **Health Monitoring** – Can be integrated into **hospital management systems**.  

---

## 🛠️ Tech Stack  

| Component       | Technology Used |
|----------------|----------------|
| **Model**      | Transfer Learning (ResNet) 🧠 |
| **Backend**    | FastAPI 🚀 |
| **Experiment Tracking** | MLflow 📊 |
| **Data Versioning** | DVC 📂 |
| **Deployment** | Docker, AWS EC2 ☁ |
| **CI/CD**      | GitHub Actions 🔄 |
| **Storage**    | AWS S3 🗄 |

---

## 📊 Results & Model Performance  
- ✅ Achieved **96% accuracy** on chest disease classification.  
- 🔍 **Reduced false negatives** to enhance diagnostic reliability.  
- 📉 **High recall** to ensure minimal missed disease cases.  
- ⚖ **Balanced F1-score** for improved precision and recall.  
- 🚀 **Optimized model performance** using transfer learning techniques.  


![Screenshot (97)](https://github.com/user-attachments/assets/0ea82693-aa45-4c1f-b9f6-3eb207382b87)





