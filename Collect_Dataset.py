import cv2
import os
import RPi.GPIO as GPIO # type: ignore
from picamera2 import Picamera2 # type: ignore
# Initialize camera
#cap = cv2.VideoCapture(0)

PICAM2 = Picamera2()
# Setup Camera configurations
PICAM2.preview_configuration.main.size = (1280, 720)
PICAM2.preview_configuration.main.format = 'RGB888'
PICAM2.preview_configuration.align()
PICAM2.configure('preview')
PICAM2.start()
# Load face cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Counter for capturing images
count = 0

# Input user's name
name = input("Enter your name: ")

# Create a directory for the user's dataset
user_dataset_dir = os.path.join("images", name)
if not os.path.exists(user_dataset_dir):
    os.makedirs(user_dataset_dir)
print("test")
while True:
    #ret, frame = cap.read()
    # Capture image from Raspberry Pi Camera
    frame = PICAM2.capture_array()
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Crop the face region
        face_roi = frame[y:y+h, x:x+w]

        # Resize the face to 100x100 pixels
        resized_face = cv2.resize(face_roi, (100, 100))

        # Save the resized face (in color)
        cv2.imwrite(f"{user_dataset_dir}/{count}.jpg", resized_face)

    # Display the image
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 400:  # Capture 400 images or press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
