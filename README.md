YOLO-Haarcascade Emotion Recognition with Real-Time Face Detection
This project implements a real-time emotion recognition system that utilizes both a custom-trained YOLOv8 model and Haarcascade face detection models. Detected faces are processed through a custom-trained Convolutional Neural Network (CNN) model to predict emotions. This project compares the performance of both YOLOv8 and Haarcascade face detection methods and overlays the predicted emotions on a live video feed.

Features
Custom YOLOv8 for Face Detection: A custom-trained YOLOv8 model specifically trained on a face dataset is used for fast and accurate face detection.
Haarcascade for Face Detection: Provides an alternative face detection method for comparison.
Emotion Recognition: A CNN model trained on the FER-2013 dataset predicts seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
Real-time Processing: Detects faces and predicts emotions in real-time using a webcam feed.
Comparison of Detection Methods: Displays results from both YOLOv8 and Haarcascade side by side to evaluate the performance of each.
Demo

Add a GIF or image demonstrating the system in action.

Installation
To get started with this project, follow the steps below:

1. Clone the Repository
First, clone this repository to your local machine:

bash
Copy code
git clone https://github.com/Shay-Ostrovsky/YOLO-Haarcascade-Emotion-Recognition-with-Real-Time-Face-Detection.git
cd YOLO-Haarcascade-Emotion-Recognition-with-Real-Time-Face-Detection
2. Set up a Virtual Environment (Optional)
It's recommended to create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Required Packages
Install the dependencies from requirements.txt:

bash
Copy code
pip install -r requirements.txt
4. Download YOLOv8 Pretrained Model
Download the custom YOLOv8 face detection model (new.pt) that was trained on a face dataset and place it in the root directory.

5. Download Haarcascade Model
Download the Haarcascade face detection model from OpenCV GitHub and place it in the root directory.

6. Run the Application
Start the real-time emotion detection:

bash
Copy code
python main.py
Ensure your webcam is connected and accessible for live video feed.

How It Works
Face Detection:

The system first uses both the custom-trained YOLOv8 and Haarcascade models to detect faces in real time.
Faces are extracted as regions of interest (ROI) from the live video feed.
Emotion Recognition:

The detected face ROIs are passed to a custom CNN model, which predicts one of seven emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).
The emotions are overlaid on the live feed for each detected face.
Side-by-Side Comparison:

The system displays the YOLOv8 and Haarcascade detection results in separate windows for performance comparison.
Dataset
The emotion recognition model is trained on the FER-2013 dataset, which contains grayscale facial images categorized into seven emotions.

Models Used
Custom YOLOv8: A custom-trained YOLOv8 model trained on a face detection dataset.
Haarcascade: A classic OpenCV face detection model.
Custom CNN: A convolutional neural network model trained on the FER-2013 dataset for emotion classification.
