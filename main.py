import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO  # Assuming YOLOv8 is from the Ultralytics package

class YOLOEmotionFaceDetector:
    def __init__(self, yolo_model_path, emotion_model_path):
        # Load the pre-trained YOLO model for face detection
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load the emotion detection model
        self.emotion_model = load_model(emotion_model_path)
        
        # Define emotion labels (based on the class indices you provided)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def detect_faces_yolo(self, frame):
        # Run YOLO face detection on the frame
        results = self.yolo_model(frame)
        
        # Extract bounding boxes from YOLO results
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

    def predict_emotion(self, face_region):
        # Resize face region to match the emotion model input size (48x48)
        resized_face = cv2.resize(face_region, (48, 48))
        
        # Normalize the face region (convert to float32 and scale between 0 and 1)
        normalized_face = resized_face.astype('float32') / 255.0
        
        # Expand dimensions to match the input shape of the emotion model (1, 48, 48, 1)
        input_face = np.expand_dims(normalized_face, axis=0)
        input_face = np.expand_dims(input_face, axis=-1)  # For grayscale images
        
        # Predict the emotion using the emotion model
        emotion_prediction = self.emotion_model.predict(input_face)
        
        # Get the index of the highest probability
        emotion_label_idx = np.argmax(emotion_prediction)
        
        # Get the corresponding emotion label
        emotion_label = self.emotion_labels[emotion_label_idx]
        
        # Also return the confidence of the predicted emotion
        confidence = np.max(emotion_prediction)
        
        return emotion_label, confidence

    def process_frame(self, frame, cascade_classifier):
        # Create a copy of the frame to show both results side by side
        yolo_frame = frame.copy()
        haar_frame = frame.copy()
        
        # Detect faces using YOLO
        yolo_faces = self.detect_faces_yolo(yolo_frame)
        haar_faces = self.detect_faces_haarcascade(haar_frame, cascade_classifier)

        # Convert the frame to grayscale for emotion detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Process YOLO-detected faces
        for (x, y, w, h) in yolo_faces:
            face_region = gray_frame[y:y+h, x:x+w]
            emotion, confidence = self.predict_emotion(face_region)
            cv2.rectangle(yolo_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(yolo_frame, f'{emotion} ({confidence*100:.2f}%)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Process Haarcascade-detected faces
        for (x, y, w, h) in haar_faces:
            face_region = gray_frame[y:y+h, x:x+w]
            emotion, confidence = self.predict_emotion(face_region)
            cv2.rectangle(haar_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(haar_frame, f'{emotion} ({confidence*100:.2f}%)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        return yolo_frame, haar_frame


# Example usage
if __name__ == "__main__":
    # Paths to the YOLO face detection model and emotion detection model
    yolo_model_path = 'new.pt'  # Replace with your trained YOLO model path
    haarcascade_path = 'haarcascade_frontalface_default.xml'
    emotion_model_path = 'try./best_model.h5'

    # Load the Haarcascade model
    haarcascade = cv2.CascadeClassifier(haarcascade_path)

    # Create an instance of YOLOEmotionFaceDetector
    detector = YOLOEmotionFaceDetector(yolo_model_path, emotion_model_path)

    # Initialize webcam video capture
    cap = cv2.VideoCapture(1)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame for face and emotion detection using YOLO and Haarcascade
        yolo_frame, haar_frame = detector.process_frame(frame, haarcascade)

        # Display the resulting frames in two separate windows
        cv2.imshow('YOLO Emotion Face Detection', yolo_frame)
        cv2.imshow('Haarcascade Emotion Face Detection', haar_frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
