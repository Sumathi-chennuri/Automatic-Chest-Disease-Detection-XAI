🫁 Automatic Chest Disease Detection Using Deep Learning & XAI

📌 Overview

This project presents an AI-powered system for automatic detection of 14 thoracic diseases from chest X-ray images using a fine-tuned ResNet50 model integrated with Explainable AI (Grad-CAM).
The system is deployed as a Streamlit web application, enabling real-time prediction and visual interpretability.

🚀 Features

Multi-label classification of 14 chest diseases

Probability-based predictions using sigmoid activation

Grad-CAM heatmaps for visual explanation

Real-time web interface using Streamlit

Clinically interpretable outputs

🧠 Model Architecture

Backbone: ResNet50 (ImageNet pre-trained)

Output Layer: 14 neurons (multi-label)

Loss Function: Binary Cross-Entropy

Explainability: Grad-CAM on last convolutional layer

🛠️ Tech Stack

Python

PyTorch

Streamlit

Grad-CAM

NumPy, PIL, Matplotlib
