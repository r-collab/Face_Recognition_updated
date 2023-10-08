import dlib
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC

# Load the pre-trained face detector from dlib
detector = dlib.get_frontal_face_detector()

# Load the pre-trained facial landmarks model from dlib
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the pre-trained face recognition model from dlib
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load the pre-trained SVM model for face identification
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)


# Function to extract 128-dimensional features from a face
def extract_face_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Iterate over detected faces
    face_features = []
    for face in faces:
        # Predict facial landmarks for the detected face
        shape = predictor(gray, face)

        # Calculate the 128-dimensional face features
        features = np.array(facerec.compute_face_descriptor(image, shape))
        face_features.append(features)

    return faces, face_features


# Function to identify faces using the SVM model
def identify_faces(face_features):
    # Prepare an empty list to store the predicted labels
    labels = []

    # Iterate over the face features
    for features in face_features:
        # Reshape the features for prediction
        features = features.reshape(1, -1)

        # Predict the label using the SVM model
        label = svm_model.predict(features)[0]

        # Append the predicted label to the list
        labels.append(label)

    return labels


# Capture video from the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    if not ret:
        # If the frame is empty, break the loop
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract facial features from the frame
    faces, face_features = extract_face_features(frame)

    if len(faces) > 0:
        # Identify the faces using the SVM model
        labels = identify_faces(face_features)

        # Iterate over the detected faces and their corresponding labels
        for face, label in zip(faces, labels):
            # Extract the coordinates of the bounding box
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the name label below the bounding box
            if label == 'unknown':
                label_text = 'Unknown'
            else:
                label_text = label
            cv2.putText(frame, label_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Identification', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
