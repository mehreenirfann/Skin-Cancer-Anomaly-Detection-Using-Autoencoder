# Skin-Cancer-Anomaly-Detection-Using-Autoencoder
# Anomaly Detection for Malignant Skin Cancer Diagnosis
**Course:** EEE 443 - Project  
**Team:** Mehreen Irfan & Yusuf Karacalı  

## 📌 Overview
This repository contains our unsupervised Deep Learning framework for diagnosing malignant skin cancer. Traditional supervised models struggle with the extreme scarcity of malignant training data. To solve this class imbalance, our Convolutional Autoencoder (CAE) is trained *exclusively* on benign (normal) skin lesions from the ISIC dataset. 

When a malignant lesion is processed, the model fails to accurately reconstruct its chaotic and irregular features. This results in a high reconstruction error, which the system uses to successfully flag the lesion as an anomaly.

## 🚀 Key Results
Tested on a separate dataset of unseen benign and malignant images, the model achieved:
* **Sensitivity (Malignant Detection):** 65.62%
* **Specificity (Benign Validation):** 57.41%

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** TensorFlow, Keras, OpenCV (for morphological Black-Hat hair removal)

## 📄 Full Methodology
For a deep dive into the neural network architecture, our hybrid MSE-SSIM loss function, statistical thresholding math, and detailed error heatmaps, please read our full **[Final Report (PDF)](Report.pdf)** included in this repository.
