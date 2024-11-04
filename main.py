import cv2
import numpy as np
from tensorflow.keras.models import load_model
<<<<<<< HEAD
from ultralytics import YOLO  # Assuming YOLOv8 is from the Ultralytics package
import os
=======
from ultralytics import YOLO  
>>>>>>> b8ddc3363c1ad84830bde69a0443bf5d4a85b2af

class YOLOEmotionFaceDetector:
    def __init__(self, yolo_model_path, emotion_model_path):
        # Load the pre-trained YOLO model for face detection
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load the emotion detection model
        self.emotion_model = load_model(emotion_model_path)
        
        # Define emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def detect_faces_yolo(self, frame):
        # Run YOLO face detection on the frame
        results = self.yolo_model(frame)
        
        # Extract bounding boxes from YOLO results:
        faces = []
        for result in results:
            for bbox in result.boxes.xyxy:  # YOLO boxes are in the format [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, bbox)  # Convert to integer values
                faces.append((x1, y1, x2 - x1, y2 - y1))  # Convert to (x, y, w, h) format
        
        return faces

    def detect_faces_haarcascade(self, frame, cascade_classifier):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def predict_emotion(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        predictions = self.emotion_model.predict(face_img)
        emotion_index = np.argmax(predictions)
        return self.emotion_labels[emotion_index]

    def process_frame(self, frame, cascade_classifier):
        yolo_faces = self.detect_faces_yolo(frame)
        haar_faces = self.detect_faces_haarcascade(frame, cascade_classifier)
        
        # Draw bounding boxes for YOLO faces
        yolo_frame = frame.copy()
        for (x, y, w, h) in yolo_faces:
            cv2.rectangle(yolo_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw bounding boxes for Haarcascade faces
        haar_frame = frame.copy()
        for (x, y, w, h) in haar_faces:
            cv2.rectangle(haar_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return yolo_frame, haar_frame

<<<<<<< HEAD
# Example usage
if __name__ == "__main__":
    # Paths to the YOLO face detection model and emotion detection model
    yolo_model_path = 'C:\\Git\\Video\\Emotion_Recognition_wReal_Time_Face_Detection\\RealTimeFaceEmotionDetection\\YOLO-Haarcascade-Emotion-Recognition-with-Real-Time-Face-Detection\\yolo_face_detection.pt'  # Replace with your trained YOLO model path
    haarcascade_path = 'haarcascade_frontalface_default.xml'
    emotion_model_path = 'C:\\Git\\Video\\Emotion_Recognition_wReal_Time_Face_Detection\\RealTimeFaceEmotionDetection\\YOLO-Haarcascade-Emotion-Recognition-with-Real-Time-Face-Detection\\emotion_regulation_model.h5'
=======

if __name__ == "__main__":
    # Paths to the YOLO face detection model and emotion detection model
    yolo_model_path = 'yolo_face_detection.pt'
    haarcascade_path = 'haarcascade_frontalface_default.xml'
    emotion_model_path = 'emotion_regulation_model.h5'
>>>>>>> b8ddc3363c1ad84830bde69a0443bf5d4a85b2af

    # Load the Haarcascade model
    haarcascade = cv2.CascadeClassifier(haarcascade_path)

    # Create an instance of YOLOEmotionFaceDetector
    detector = YOLOEmotionFaceDetector(yolo_model_path, emotion_model_path)

    # Initialize webcam video capture
    cap = cv2.VideoCapture(0)
<<<<<<< HEAD

    # Variables for fall detection
    previous_y = None
    fall_threshold = 50  # Adjust this threshold based on your requirements
=======
>>>>>>> b8ddc3363c1ad84830bde69a0443bf5d4a85b2af

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame for face and emotion detection using YOLO and Haarcascade
        yolo_frame, haar_frame = detector.process_frame(frame, haarcascade)

        # Fall detection logic
        faces = detector.detect_faces_yolo(frame)
        if faces:
            x, y, w, h = faces[0]  # Assuming the first detected face
            face_img = frame[y:y+h, x:x+w]
            emotion = detector.predict_emotion(face_img)
            cv2.putText(yolo_frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            if previous_y is not None and abs(previous_y - y) > fall_threshold:
                cv2.putText(yolo_frame, "Fall Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            previous_y = y

        # Display the resulting frames in two separate windows
        cv2.imshow('YOLO Emotion Face Detection', yolo_frame)
        cv2.imshow('Haarcascade Emotion Face Detection', haar_frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()