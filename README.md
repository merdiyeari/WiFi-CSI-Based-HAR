# WiFi CSI-Based Human Activity Recognition (HAR)

[🇹🇷 **Türkçe Özet:** Bu proje, ortamdaki mevcut WiFi sinyallerini (CSI) birer sensör gibi kullanarak insan hareketlerini (Yatma, Düşme, Yürüme, Bir Şey Alma, Koşma, Oturma, Ayağa Kalkma) tespit eden **1D-CNN + LSTM** tabanlı bir derin öğrenme modeli ve **Streamlit** web arayüzü sunmaktadır.]

---

This repository features a hybrid deep learning architecture and an interactive web application designed to recognize human activities using ambient WiFi Channel State Information (CSI) as passive sensors. This **device-free**, **cost-effective**, and **privacy-preserving** approach eliminates the need for camera vision systems or wearable sensors.

---

## 📌 Key Features

- **Privacy-Preserving & Light-Independent:** Operating without camera visual feeds, it ensures total user privacy and functions seamlessly regardless of lighting conditions (e.g., in complete darkness).
- **7 Core Activity Classification:** Accurately classifies Lying, Falling, Walking, Picking Up, Running, Sitting, and Standing Up.
- **Hybrid Deep Learning Architecture:** Combines **1D-CNN** (for spatial signal feature extraction) and **LSTM** (for learning temporal sequence dynamics).
- **Advanced Preprocessing:** 
  - **MinMaxScaler:** Normalizes signal amplitudes to stabilize gradient descent.
  - **PCA (Principal Component Analysis):** Reduces 90-channel noisy CSI subcarrier data down to its 10 most dominant principal components while retaining over 90% of variance.
- **Interactive Web UI:** Built with **Streamlit** to enable `.npy` signal file uploads, real-time activity predictions, confidence score displays, and dynamic signal visualizations.

---

## 📊 Dataset & Performance

- **Dataset:** UT-HAR (*University of Texas - Human Activity Recognition*)
- **Data Dimension:** 3,977 training instances across 90 subcarrier frequencies and 250 time steps.
- **Model Accuracy:** Achieved **>90% overall test accuracy**. Implemented **Early Stopping** to mitigate overfitting and maximize real-world generalization.

---

## 💾 Dataset Download

Due to GitHub file size limitations for large signal files, the test dataset is hosted externally on Kaggle:

👉 **[Download full UT-HAR Test Dataset on Kaggle](https://www.kaggle.com/datasets/hylanj/wifi-csi-dataset-ut-har)**

