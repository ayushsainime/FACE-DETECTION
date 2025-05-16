# FaceTracker

A real-time face detection and localization system built with TensorFlow, Keras, and OpenCV.  
It performs **binary classification** (face / no-face) and **bounding-box regression** (locates the face) in one end-to-end model.

---

## 📖 Overview

FaceTracker is a multi-task CNN that:
1. **Classifies** whether a face is present in a 120×120 RGB image.  
2. **Regresses** the face’s bounding box (`[x_min, y_min, x_max, y_max]` in normalized coordinates).  

We leverage a pre-trained **VGG16** feature extractor, then branch into:
- A **classification head** (sigmoid output)  
- A **regression head** (4-D sigmoid output)  

---

## ✨ Features

- **Data Augmentation** with [Albumentations](https://github.com/albumentations-team/albumentations): random crops, flips, brightness/contrast, RGB shifts, gamma.  
- **Custom Training Loop** via a subclassed `tf.keras.Model`, combining classification & localization losses.  
- **Real-time Webcam Demo** using OpenCV for on-screen bounding boxes and labels.  
- **Modular Code**: Separate data pipeline, model builder, custom loss, and inference script.  

---

## 🛠 Libraries & Tools

- **TensorFlow 2.x / Keras** – model definition, training  
- **OpenCV (cv2)** – video capture, drawing, image I/O  
- **Albumentations** – fast, flexible image augmentation  
- **NumPy** – array operations  
- **Matplotlib** – debugging & visualization  
- **Python 3.8+**  

---
## NEURAL NETWORK - VGG16 
![Screenshot 2025-05-16 135242](https://github.com/user-attachments/assets/748a3755-468a-46eb-92b9-5c8cd77d0c07)

## ALTERED MODEL TO OUR NEEDS 

![Screenshot 2025-05-16 135253](https://github.com/user-attachments/assets/c7a24b72-be11-4c45-a6fa-e293823ce76b)

## CLASSIFICATION LOSS  ,  REGRESSION LOSS  AND TOTAL LOSS 
![Screenshot 2025-05-16 135230](https://github.com/user-attachments/assets/c4416c49-f38a-490b-83b2-4b8f4f17172f)

## PREDICTION ( MAKING BOUNDING BOX AND IDENTIFYING FACE PRESENT OR NOT ) 
![Screenshot 2025-05-16 135219](https://github.com/user-attachments/assets/72ee929e-f120-4071-8b74-cbe1d1ebdb78)
![Screenshot 2025-05-16 151935(1)](https://github.com/user-attachments/assets/7949ce69-8f0d-479b-aaa4-16049f2ee2e0)



---
## SOME MORE INFORMATION 
Model Architecture
Input: 120×120×3 RGB image

Backbone: VGG16 (no top)

Classification Head:

GlobalMaxPooling2D → Dense(2048, ReLU) → Dense(1, Sigmoid)

Regression Head:

GlobalMaxPooling2D → Dense(2048, ReLU) → Dense(4, Sigmoid)

Outputs:

prob_face ∈ [0,1]

bbox_norm ⊂ [0,1]⁴


----
## USE CASE : 

Security & Surveillance: quickly locate faces in video streams.

Human–Computer Interaction: trigger actions when a face appears.

Data Labeling: semi-automated bounding-box pre-labeling.

Educational: demonstrates multi-task learning with Keras Functional API

## ⚙️ Installation

1. Clone this repo:  
   ```bash
   git clone https://github.com/your-username/FaceTracker.git
   cd FaceTracker
