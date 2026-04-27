# 🧠 Gait Recognition System (GEI + CNN + LSTM)

## 📌 Overview

This project implements a **Gait Recognition System** that identifies individuals based on their walking patterns using:

* Silhouette extraction (MediaPipe)
* Gait Energy Image (GEI)
* Deep Learning (CNN + LSTM)

---

## 🚀 Features

* Automatic silhouette extraction from videos
* Multi-person gait dataset handling
* GEI-based feature representation
* CNN + LSTM model for identification
* Real-time prediction using webcam

---

## 📂 Project Structure

```
src/                → core code
models/             → trained model
Dataset_Gait/       → dataset (excluded from repo)
demo/               → demo visuals
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Train Model

```bash
python src/gait_gei.py
```

### 2. Real-Time Prediction

```bash
python src/realtime_predict.py
```

---

## 📊 Model Architecture

* TimeDistributed CNN (feature extraction)
* LSTM (temporal modeling)
* Dense layers (classification)

---

## 🧪 Results

* Works best in controlled environments
* Accuracy depends on:

  * Dataset size
  * Silhouette quality
  * Camera consistency

---

## ⚠️ Limitations

* Sensitive to lighting and clothing
* Requires consistent camera angle
* Not suitable for real-world surveillance use

---

## 📸 Demo

![Demo](demo/demo.gif)

---

## 🧠 Future Improvements

* Use larger datasets (CASIA-B)
* Improve segmentation quality
* Add transformer-based models

---

## 👨‍💻 Author

Lokesh Modi
