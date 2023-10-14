import cv2
import time
import pygetwindow as gw
import numpy as np
from deepface import DeepFace

# Load Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cam = cv2.VideoCapture(0)

# cv2.namedWindow("Python Webcam Screenshot")

img_counter = 0

# Set the time interval for taking screenshots (in seconds)
screenshot_interval = 5

# Initialize the start time
start_time = time.time()

# Set the time threshold for displaying "Return to Work" prompt (in seconds)
no_person_threshold = 15
no_person_time = None
return_to_work_printed = False

# Set the time threshold for checking head movement (in seconds)
head_movement_threshold = 10
last_head_movement_time = time.time()

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Convert frame to grayscale for face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    eyes_detected = False  # Flag to track if eyes are currently detected

    for (x, y, w, h) in faces:
        # Region of Interest (ROI) for face and eye detection
        roi_gray = gray[y:y + h, x:x + w]

        # Detect eyes in the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) > 0:
            eyes_detected = True

        # Draw a rectangle around the detected face
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        #for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the detected eyes
            #cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        # Update the last head movement time when a face is detected
        last_head_movement_time = time.time()

    cv2.imshow("Python Webcam Screenshot", frame)

    k = cv2.waitKey(1)

    if img_counter >= 3:
        print("Maximum screenshots captured. Exiting...")
        break

    # Check if another app's window is using the camera
    if gw.getAllTitles():
        for title in gw.getAllTitles():
            if "Microsoft Teams" in title:  # Replace with the window title of Microsoft Teams
                print("Another app is using the camera. Pausing timer...")
                no_person_time = None  # Reset the timer
                break

    # Check if eyes disappeared
    if not eyes_detected:
        if no_person_time is None:
            no_person_time = time.time()
        else:
            elapsed_no_person_time = time.time() - no_person_time
            if elapsed_no_person_time >= no_person_threshold and not return_to_work_printed:
                print("Return to Work")
                return_to_work_printed = True
    else:
        no_person_time = None

    # Check for head movement
    elapsed_head_movement_time = time.time() - last_head_movement_time
    if elapsed_head_movement_time >= head_movement_threshold:
        print("Are you there?")

    # Check if it's time to capture a screenshot
    elapsed_time = time.time() - start_time
    if elapsed_time >= screenshot_interval and eyes_detected:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("Screenshot saved as {}".format(img_name))
        img_counter += 1
        start_time = time.time()  # Reset the start time for the next screenshot


# compare img
model = DeepFace.build_model("Facenet")

# Paths to the two screenshot images you want to compare
screenshot1_path = "opencv_frame_0.png"
screenshot2_path = "opencv_frame_2.png"

# Load the reference images for comparison
screenshot1 = cv2.imread(screenshot1_path)
screenshot1 = cv2.cvtColor(screenshot1, cv2.COLOR_BGR2RGB)

screenshot2 = cv2.imread(screenshot2_path)
screenshot2 = cv2.cvtColor(screenshot2, cv2.COLOR_BGR2RGB)

# Perform face verification with the two screenshots
result = DeepFace.verify(screenshot1, screenshot2, model_name="Facenet", enforce_detection=False)

# Print the verification result in Jupyter Notebook
print("Face Verification Result:", result["verified"])

cam.release()
cv2.destroyAllWindows()
