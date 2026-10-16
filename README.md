# Bollywood Face Recognition

A Bollywood celebrity face recognition system built using **OpenCV** and **Streamlit** with a limited dataset.

---

### 📌 Project Description

This project detects and recognizes Bollywood celebrities from images or webcam feed using OpenCV’s Haar Cascade classifier and LBPH (Local Binary Patterns Histograms) face recognizer.

The model is trained on a **limited dataset**, so recognition accuracy is moderate and works best with clear, frontal face images of the trained celebrities.

---

### ✨ Features

- Real-time face detection using Haar Cascade
- Face recognition using LBPH algorithm
- Interactive web interface built with Streamlit
- Displays the name of the recognized Bollywood celebrity
- Simple and beginner-friendly project

---

### 🛠️ Technologies Used

- Python
- OpenCV
- Streamlit
- NumPy

---

### 📂 Project Structure
bollywood-face_recogination/
├── app.py                 # Streamlit web application
├── faces_trains.py        # Script used for training the model
├── haar_face.xml          # Haar Cascade classifier
├── labels.npy             # Labels used during training
├── .gitignore
└── README.md


> **Note:** The trained model files (`face_trained.yml` and `features.npy`) are not included in this repository because of GitHub’s file size limit. You need to train the model yourself using `faces_trains.py`.

---

