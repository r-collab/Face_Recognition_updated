import dlib
import cv2
import pandas as pd
import numpy as np
import time
import os.path

# Load the pre-trained face detector, facial landmark predictor, and face recognition model
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Empty lists to store the face measurements and names
face_measurements = []
name = input("Enter the name of the person: ")

# Open a video capture object
cap = cv2.VideoCapture(0)

# Variables for time and counter
counter = 0

# Check if the Excel file exists
if os.path.isfile("face_measurements.xlsx"):
    # Load the existing data from the Excel file
    existing_data = pd.read_excel("face_measurements.xlsx")

    # Append the existing data to the face_measurements list
    face_measurements.extend(existing_data.values)

while True:
    # Read the frame from the video source
    ret, frame = cap.read()

    # Check if frame reading was successful
    if not ret:
        break

    # Convert the frame to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection is done
    faces = face_detector(rgb)

    # If 100 measurements are captured, break the loop
    if counter >= 100:
        break

    # If multiple faces detected, show a warning and continue to the next frame
    if len(faces) > 1:
        cv2.putText(frame, "Multiple faces detected. Please keep only one face in the frame.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Real-time Face Embedding", frame)
        continue

    # For each detected face
    for face in faces:
        # Perform facial landmark detection
        landmarks = landmark_predictor(rgb, face)

        # Calculate face embeddings
        face_descriptor = face_recognition_model.compute_face_descriptor(rgb, landmarks)

        # Convert face descriptor to a numpy array
        face_embeddings = np.append(face_descriptor, name)  # Append the name to the face measurements

        # Append the face measurements to the list
        face_measurements.append(face_embeddings)

        # Show frame with bounding box around the detected face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Increment the counter
        counter += 1

    # Show frame
    cv2.imshow("Real-time Face Embedding", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Create DataFrame for face measurements
data = pd.DataFrame(face_measurements, columns=[f"Feature {i + 1}" for i in range(128)] + ["Name"])

# Save the updated DataFrame to the Excel file
data.to_excel("face_measurements.xlsx", index=False)

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
