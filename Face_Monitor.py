import RPi.GPIO as GPIO
import time
import os
import cv2
import numpy as np
from threading import Thread
from keras_facenet import FaceNet
import pickle
from picamera2 import Picamera2

# Set GPIO mode (BCM mode)
GPIO.setmode(GPIO.BCM)

# Define GPIO pins for trigger and echo
TRIG = 23
ECHO = 24

# Set up GPIO pins
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Maximum distance to measure (in centimeters)
MAX_DISTANCE = 15

# Initialize OpenCV face detection using Haar Cascade
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Load FaceNet model
face_net = FaceNet()

# Load face database
with open("data.pkl", "rb") as file:
    database = pickle.load(file)

# Directory to save unknown face images
unknown_faces_dir = "unknown_faces"
os.makedirs(unknown_faces_dir, exist_ok=True)

# Function to process each frame
def process_frame(frame, sensor_reading):
    # Display Face Recognition status based on sensor reading
    status_text = "Face Recognition: ON" if sensor_reading else "Face Recognition: OFF"
    text_color = (0, 255, 0) if sensor_reading else (0, 0, 255)
    cv2.putText(frame, status_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
    
    # If Face Recognition is OFF, return without detecting faces
    if not sensor_reading:
        return
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        face_img = cv2.resize(frame[y:y+h, x:x+w], (160, 160))
        face_signature = face_net.embeddings(np.expand_dims(face_img, axis=0))
        
        min_dist = 0.94
        identity = 'Unknown'
        
        # Check distance to known faces in the database
        for key, value in database.items():
            dist = np.linalg.norm(value - face_signature)
            if dist < min_dist:
                min_dist = dist
                identity = key
        
        # Draw rectangle around the face
        color = (0, 0, 255) if identity == 'Unknown' else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, identity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Save frame if unknown face detected
        # if identity == 'Unknown':
        #     unknown_face_filename = os.path.join(unknown_faces_dir, f"unknown_face_{time.strftime('%Y%m%d%H%M%S')}.jpg")
        #     cv2.imwrite(unknown_face_filename, frame)
        #     print("Unknown face captured and saved.")

# Function to read frames from the camera
def read_frames(picam):
    try:
        # Setup Camera configurations
        picam.preview_configuration.main.size = (1280, 720)
        picam.preview_configuration.main.format = 'RGB888'
        picam.preview_configuration.align()
        picam.configure('preview')
        
        picam.start()
        while True:
            # Capture frame from Picamera2
            frame = picam.capture_array()
            
            current_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, current_datetime, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Read sensor reading
            sensor_reading = read_sensor()
            
            # Process every frame
            process_frame(frame, sensor_reading)
            
            # Display the resulting frame
            cv2.imshow('frame', frame)
            
            # Break the loop on 'ESC' key press
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        picam.stop()
        # Close windows
        cv2.destroyAllWindows()

# Function to read sensor
def read_sensor():
    # Set trigger to HIGH for 10 microseconds
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    # Measure the duration for which the echo pin is HIGH
    pulse_start = time.time()
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    pulse_end = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start

    # Calculate distance (in centimeters)
    distance = pulse_duration * 17150

    # Return True if distance is less than MAX_DISTANCE, False otherwise
    return distance < MAX_DISTANCE

# Main function
def main():
    picam = Picamera2()
    
    # Create and start thread for reading frames
    frame_thread = Thread(target=read_frames, args=(picam,))
    frame_thread.start()

    # Wait for the frame thread to finish
    frame_thread.join()

if __name__ == "__main__":
    main()
