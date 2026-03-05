# 🫁 Automatic Chest Disease Detection Using Deep Learning & XAI

## 📌 Overview

This project presents an AI-powered system for automatic detection of 14 thoracic diseases from chest X-ray images using a fine-tuned ResNet50 deep learning model integrated with Explainable Artificial Intelligence (XAI) through Grad-CAM.

The system is deployed as a Streamlit web application, enabling users to upload chest X-ray images, obtain multi-label disease predictions with probability scores, and visualize heatmaps highlighting clinically relevant regions that influenced the model’s decisions.

## 🚀 Key Features

Multi-label classification of 14 thoracic diseases

Sigmoid-based probability outputs for each disease

Grad-CAM heatmaps for explainable predictions

Real-time inference via Streamlit web interface

User-friendly and clinically interpretable outputs

## 🧠 Model Architecture

Backbone: ResNet50 (ImageNet pre-trained)

Fine-tuning: Final fully connected layer modified for 14 disease classes

Activation Function: Sigmoid (multi-label classification)

Loss Function: Binary Cross-Entropy (BCE)

Explainability Module: Grad-CAM applied to the final convolutional layer

## 🛠️ Tech Stack

Programming Language: Python

Deep Learning Framework: PyTorch

Web Framework: Streamlit

Explainable AI: Grad-CAM

Image Processing & Visualization: NumPy, PIL, Matplotlib

## ▶️ Running the Application

The Streamlit web application is launched using the following command:

```bash

## 📂 Project Structure

Automatic-Chest-Disease-Detection-XAI/
│
├── src/                      # Streamlit app and model logic
│   └── app.py
│
├── docs/                     # Documentation and outputs
│   ├── Report.pdf
│   ├── results.png
│   └── input.png
│
├── architecture.png          # System architecture diagram
├── requirements.txt          # Project dependencies
├── demo_video_link.txt       # Google Drive demo video link
├── setup_instructions.md     # Steps to run the project
└── README.md

