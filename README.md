# armbharatchallenge
PS 3- Real-Time Road Anomaly Detection from Dashcam Footage on Raspberry Pi
# 🚗 Road AI – Real-Time Embedded Road Safety System

Road AI is an embedded AI-based road monitoring and driver assistance system built using Raspberry Pi 5 and a custom-trained YOLO object detection model.

The system detects potholes and multiple road obstacles in real-time and provides visual and audio alerts using LEDs, a buzzer, and an OLED display.

---

## 📌 Project Overview

This system detects the following six classes:

- 🕳 POTHOLE  
- 🚶 PERSON  
- 🐕 ANIMAL  
- 🚗 VEHICLE  
- 🧱 OBSTACLE  
- 🛣 CRACK  

The primary objective is accurate real-time pothole detection, while also identifying other common road hazards.

The final deployed model is a fully quantized INT8 TensorFlow Lite model optimized for ARM architecture and Raspberry Pi 5.

---

## 🧠 AI Model Details

- Architecture: YOLO (Ultralytics)
- Training Platform: Kaggle Tesla P100 GPU
- Dataset Size: ~55,000 labeled images
- Epochs: 100+
- Number of Classes: 6
- Validation Images: 10,959
- mAP50: 0.72
- mAP50-95: 0.446
- Final Model Format: INT8 TensorFlow Lite
- Final Model Size: ~2.7 MB

### Model Conversion Pipeline

best.pt → ONNX → TensorFlow SavedModel → INT8 TFLite

This structured conversion ensured optimized performance on Raspberry Pi while maintaining detection accuracy.

---

## 🛠 Hardware Components

- Raspberry Pi 5  
- USB Webcam  
- 6 Class-Specific LEDs  
- Buzzer (activated for potholes)  
- SSD1306 OLED Display  
- GPIO Interface Wiring  

Each class is mapped to a dedicated LED.  
The bounding box color on the screen matches the LED color for intuitive visual feedback.

---

## ⚙️ Key Features

- Real-time multi-class object detection  
- Class-specific LED alerts  
- Buzzer alert for pothole detection  
- OLED display showing detected class and confidence  
- Bounding box smoothing to reduce flickering  
- Confidence smoothing for stable display values  
- Timestamped image logging for high-confidence detections  
- FPS and CPU temperature monitoring  
- Multi-threaded architecture for stable performance  
- INT8 quantized model optimized for ARM processors  

---

## 🚀 System Workflow

1. Capture live frame from USB webcam  
2. Run inference using INT8 TFLite model  
3. Process detections and smooth bounding boxes  
4. Activate corresponding LED  
5. Trigger buzzer for potholes  
6. Update OLED display  
7. Log high-confidence detections  
8. Display annotated video feed  

---

## 📂 Project Structure
Road-AI/
│
├── Training Results/
│ └── (All training data, metrics, logs, and weight files)
│
├── Model Conversion/
│ └── (best.pt, ONNX model, SavedModel, INT8 TFLite model)
│
├── Example Logs/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── detections.csv
│
├── Source Code.py
├── Report.pdf
├── README.md
└── documentation.txt


---

## 📊 Performance Summary

- Stable real-time inference on Raspberry Pi 5  
- ARM-optimized INT8 deployment  
- Consistent FPS during operation  
- Controlled thermal performance  
- Accurate multi-class detection across varied road scenarios  

---

## 🎯 Project Objectives Achieved

- Real-time pothole detection  
- Multi-class road obstacle detection  
- Embedded AI deployment on Raspberry Pi 5  
- Hardware-integrated visual and audio alert system  
- Performance optimization for edge computing  
- Stable and reliable real-time operation  

---

## 📜 License

This project was developed for academic and research purposes.