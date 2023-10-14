import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 represents the default camera (change if needed)

# Initialize background frame
background = None

# Set a threshold for detecting motion
motion_threshold = 35000 # You may need to adjust this value based on your environment

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for motion analysis
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if background is None:
        background = gray_frame
        continue

    # Calculate the absolute difference between the current frame and the background
    frame_delta = cv2.absdiff(background, gray_frame)
    _, thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > motion_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

    # Show the current frame
    cv2.imshow("Video Feed", frame)

    if motion_detected:
        print("Motion detected")
    else:
        print("Static image")

    # Exit the program when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()