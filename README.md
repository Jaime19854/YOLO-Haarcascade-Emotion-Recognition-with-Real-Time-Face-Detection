# **YOLO-Haarcascade Emotion Recognition with Real-Time Face Detection**

This project implements a real-time emotion recognition system that uses both a **custom-trained YOLOv8** model and the **Haarcascade** face detection model. Detected faces are processed by a custom **Convolutional Neural Network (CNN)** model to predict emotions. The project compares the performance of YOLOv8 and Haarcascade for face detection and displays the predicted emotions on a live video feed.

## **Features**

- **Custom YOLOv8 for Face Detection**  
  Utilizes a custom-trained YOLOv8 model for accurate, fast face detection.
  
- **Haarcascade for Face Detection**  
  Implements Haarcascade face detection for performance comparison.

- **Emotion Recognition**  
  Predicts seven emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) using a CNN model trained on the FER-2013 dataset.

- **Real-time Processing**  
  Detects faces and predicts emotions from live webcam video feed in real time.

- **Comparison of Detection Methods**  
  Displays both YOLOv8 and Haarcascade face detection results side by side for evaluation.

## **Demo**
![Demo GIF](demo1.jpg)  
*Add a GIF or image to demonstrate the system in action.*

## **Installation**

Follow these steps to set up and run the project:

### **1. Clone the Repository**
```bash
git clone https://github.com/Shay-Ostrovsky/YOLO-Haarcascade-Emotion-Recognition-with-Real-Time-Face-Detection.git
cd YOLO-Haarcascade-Emotion-Recognition-with-Real-Time-Face-Detection
```

### **2. Set up a Virtual Environment (Optional)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Required Packages**
```bash
pip install -r requirements.txt
```

### **4. Download YOLOv8 Pretrained Model**
Download the custom-trained YOLOv8 face detection model (`new.pt`) and place it in the root directory.

### **5. Download Haarcascade Model**
Download the Haarcascade model from [OpenCV GitHub](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) and place it in the root directory.

### **6. Run the Application**
```bash
python main.py
```
Ensure your webcam is connected for real-time video feed.

## **How It Works**

1. **Face Detection:**  
   Both YOLOv8 and Haarcascade detect faces in the live video feed. Faces are extracted as regions of interest (ROI).

2. **Emotion Recognition:**  
   The detected face ROIs are fed into the CNN model, which predicts one of the seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

3. **Side-by-Side Comparison:**  
   The results from both YOLOv8 and Haarcascade are shown in separate windows for performance evaluation.

## **Dataset**

The CNN model is trained on the **FER-2013** dataset, which contains grayscale facial images categorized into seven emotions.

## **Models Used**

- **Custom YOLOv8:**  
  Trained specifically on a face detection dataset.
  
- **Haarcascade:**  
  OpenCV’s classic face detection model.
  
- **Custom CNN:**  
  Trained on the FER-2013 dataset for emotion classification.

## **Requirements**

- Python 3.8+
- TensorFlow 2.x
- OpenCV 4.x
- Ultralytics YOLOv8
- Dependencies listed in `requirements.txt`

## **Contributing**

Contributions are welcome! Feel free to submit issues or pull requests. Any suggestions or improvements are greatly appreciated.

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
