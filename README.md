# 🛰️ Space Station Object Detection  
### 💡 Hacksetu 1.0  
**Team:** CogniCore 


## 🧩 Overview

This project was developed for the **Hacksetu 1.0** hosted by **Amity University, Gwalior**. We utilized **YOLOv8** and synthetic data from Duality AI’s **Falcon platform** to train an object detection model capable of identifying essential tools in a simulated zero-gravity space station environment.  

### 🎯 Objects Detected:
- 🔧 **Toolbox**  
- 🧯 **Fire Extinguisher**  
- 🛢️ **Oxygen Tank**  


## 🛠️ Tech Stack

- **Model:** YOLOv8  
- **Frameworks:** PyTorch, Ultralytics  
- **Tools:** OpenCV, Matplotlib  
- **Data Source:** Falcon Simulation Platform (YOLO format)  

## 🗂️ Repository Structure

![Screenshot 2025-06-24 091202](https://github.com/user-attachments/assets/d65037b5-39d6-4fbb-9897-4976368a4759)

## 🚀 Getting Started

### 1️⃣ Create Environment

Install [Anaconda](https://www.anaconda.com/products/distribution) and run the following commands:

- conda create -n EDU python=3.9 -y
- conda activate EDU
- pip install ultralytics opencv-python matplotlib torch torchvision torchaudio


### 2️⃣ Training the Model


- python train.py


- This will begin YOLOv8 training using the synthetic Falcon dataset.

### 3️⃣ Run Inference & Evaluate


python predict.py

Generates:

* 🔹 mAP\@0.5 and mAP\@0.5:0.95
* 🔹 Precision-Recall Curve
* 🔹 F1 vs Confidence Curve
* 🔹 Sample Predictions
* 🔹 (Optional) Confusion Matrix


## 📊 Performance Metrics

| Metric        | Value |
| ------------- | ----- |
| **mAP\@0.5**  | 0.993 |
| **Precision** | 0.99  |
| **Recall**    | 0.91  |
| **F1 Score**  | 0.94  |

### 🔍 Class-wise Detection Accuracy

* ✅ **Fire Extinguisher**: 0.962
* ✅ **Toolbox**: 0.936
* ✅ **Oxygen Tank**: 0.925


## 🏁 Conclusion

This solution demonstrates the effectiveness of combining synthetic environments with real-time detection models like YOLOv8 for mission-critical operations in constrained environments such as space stations.

> Designed, implemented, and documented with precision — **Team Tech Titans, CodeClash 2.0*
