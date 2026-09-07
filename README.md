# Deepfake Video Detection System

Team project — CNN-based deepfake video detection system built using the FaceForensics++ dataset.

## Overview
This project detects deepfake videos by extracting faces from video frames and classifying them using pre-trained CNN architectures. As a team, we benchmarked three models — ResNet50, VGG19, and DenseNet121 — to compare detection accuracy.

## Dataset
- **FaceForensics++** — 7,000 video samples
- Over 10,000 video frames processed for face extraction

## My Contribution
- Built the video preprocessing pipeline using OpenCV
- Implemented face detection and extraction using MTCNN
- Trained and evaluated the DenseNet121 model as part of the 3-model comparison

## Tech Stack
Python, OpenCV, MTCNN, TensorFlow/Keras, CNNs (ResNet50, VGG19, DenseNet121)

## Notes
This was a collaborative team project. The final web deployment (Flask app) was built by another team member and is not included in this repository.
