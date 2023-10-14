# with print statements


import customtkinter
import threading
import cv2
import time
from plyer import notification
import pygetwindow as gw
import numpy as np
from deepface import DeepFace
import keyboard

temp = 0
# Create a global variable to track user input
user_input = {"number1": 20*60, "number2": 20}
exit_loop= False

show_notifications = True
# Function for scheduling notifications
def schedule_notifications():
    global user_input,show_notifications
    notification.notify(
        title="Time Set",
        message="You will be reminded",
        timeout=5
    )
# Create a thread for notification scheduling
notification_thread = threading.Thread(target=schedule_notifications)
notification_thread.daemon = True  # Allow the thread to exit when the main program ends

# Function for the GUI and user input
def run_gui():
    global user_input, notification_thread,entry_1,entry_2,frame_1
    customtkinter.set_appearance_mode("System")
    customtkinter.set_default_color_theme("blue")

    app = customtkinter.CTk()
    app.geometry("400x400")
    app.title("Notification example")

    frame_1 = customtkinter.CTkFrame(master=app)
    frame_1.pack(pady=20, padx=20, fill="both", expand=True)

    label_1 = customtkinter.CTkLabel(master=frame_1, justify=customtkinter.LEFT, text="Notify Me")
    label_1.pack(pady=10, padx=10)

    entry_1 = customtkinter.CTkEntry(master=frame_1, placeholder_text="Enter working time in sec")
    entry_1.pack(pady=10, padx=10)

    entry_2 = customtkinter.CTkEntry(master=frame_1, placeholder_text="Enter break time in sec")
    entry_2.pack(pady=10, padx=10)

    def save_number():
        global user_input, notification_thread
        try:
            # Get the text entered into entry_1 and entry_2 and convert them to integers
            user_input["number1"] = int(entry_1.get()) if entry_1.get() else (20*60)
            user_input["number2"] = int(entry_2.get()) if entry_2.get() else 20
            if not notification_thread.is_alive():
                notification_thread = threading.Thread(target=schedule_notifications)
                notification_thread.daemon = True
                notification_thread.start()
            app.destroy()
        except ValueError:
            pass

    save_button = customtkinter.CTkButton(master=frame_1, text="Save Time", command=save_number)
    save_button.pack(pady=10, padx=10)

    app.mainloop()

def verify_images(image_path1, image_path2):
    # Load the pre-trained model for face recognition
    model = DeepFace.build_model("Facenet")

    # Load the reference images for comparison
    screenshot1 = cv2.imread(image_path1)
    screenshot1 = cv2.cvtColor(screenshot1, cv2.COLOR_BGR2RGB)

    screenshot2 = cv2.imread(image_path2)
    screenshot2 = cv2.cvtColor(screenshot2, cv2.COLOR_BGR2RGB)

    # Perform face verification with the two images
    result = DeepFace.verify(screenshot1, screenshot2, model_name="Facenet", enforce_detection=False)

    return result["verified"]


# Function for webcam monitoring and face verification
def webcam_monitoring():
    global show_notifications,temp
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cam = cv2.VideoCapture(0)

# cv2.namedWindow("Python Webcam Screenshot")

    img_counter = 0

    # Set the time interval for taking screenshots (in seconds)
    screenshot_interval = 30

    # Initialize the start time
    start_time = time.time()

    # Set the time threshold for displaying "Return to Work" prompt (in seconds)
    no_person_threshold = 15
    no_person_time = None
    return_to_work_printed = False

    # Set the time threshold for checking head movement (in seconds)
    head_movement_threshold = 10
    last_head_movement_time = time.time()

    paused = False
    capture_allowed = True
    timer_started = None
    consecutive_false_count = 0
    last_true_image = None
    person_start_time = None
    person_present = False
    image_counter=0

    while True:
        if not paused:
            ret, frame = cam.read()
            
            if not ret:
                #print("Failed to capture frame")

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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                for (ex, ey, ew, eh) in eyes:
                    # Draw a rectangle around the detected eyes
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

                # Update the last head movement time when a face is detected
                last_head_movement_time = time.time()

            # cv2.imshow("Python Webcam Screenshot", frame)

            k = cv2.waitKey(1)
            
            elapsed_no_person_time = 0

            if img_counter >= ((user_input["number1"]/screenshot_interval)+1):
                #print("Maximum screenshots captured. Exiting...")
                temp = 1
                notification.notify(
                    title="Take A Break",
                    message="Rest your eyes and look at a far away object",
                    app_icon = r'C:\Users\Vedanshi\OneDrive\Desktop\dell\break.ico',
                    timeout=user_input["number2"]
                )
                time.sleep(user_input["number2"])
                break

            # Check if another app's window is using the camera

            # Check if eyes disappeared
            if not eyes_detected and capture_allowed:
                if no_person_time is None:
                    no_person_time = time.time()
                else:
                    elapsed_no_person_time = time.time() - no_person_time
                    if elapsed_no_person_time >= no_person_threshold and not return_to_work_printed:
                        print("Return to Work")
                        return_to_work_printed = True

                # If no one is detected for 30 seconds, stop capturing
                if elapsed_no_person_time >= 30:
                    capture_allowed = False
                    timer_started = None
                    #print("No one is there for 30 seconds. Stopping capture.")
                    notification.notify(
                    title="No Face infront of screen",
                    message="Camera paused look into the screen to continue",
                    app_icon = r'C:\Users\Vedanshi\OneDrive\Desktop\dell\break.ico',
                    timeout=5
                )

            # If someone is detected, reset the timer
            if eyes_detected:
                no_person_time = None
                if timer_started is None:
                    timer_started = time.time()
                    # print("Timer started.")
                    temp=1
                    
                    capture_allowed = True

            # Check for head movement
            elapsed_head_movement_time = time.time() - last_head_movement_time
            #if elapsed_head_movement_time >= head_movement_threshold:
                #print("Are you there?")

            elapsed_time = time.time() - start_time

            # If the timer is running and 30 seconds have passed, stop capturing
            #if timer_started is not None and elapsed_time - timer_started >= 30:
                #capture_allowed = False
                #timer_started = None
                #print("No one is there for 30 seconds. Stopping capture.")

            if elapsed_time >= screenshot_interval and eyes_detected and capture_allowed:
                verification_result_str = True
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("Screenshot saved as {}".format(img_name))
                
                if img_counter > 0:
                    if verification_result_str:
                        if last_true_image is not None:
                            previous_image = last_true_image  # Compare with the last "true" image
                        else:
                            previous_image = "opencv_frame_{}.png".format(img_counter - 1)  # Compare with the previous image

                    current_image = img_name
                    verification_result = verify_images(previous_image, current_image)
                    verification_result_str = "True" if verification_result else "False"
                    print("Verification Result for {}: {}: {}".format(previous_image, current_image, verification_result_str))

                    if verification_result:
                        consecutive_false_count = 0  # Reset the counter if result is True
                        last_true_image = current_image  # Update the last "true" image
                    else:
                        consecutive_false_count += 1

                    if consecutive_false_count >= 3:
                        temp = 0
                        notification.notify(
                            title="New User detected",
                            message="Restarting the program",
                            #app_icon = r'D:\\coding\\Python\\break.ico',
                            timeout=5
                        )
                        last_true_image = None  # Reset the last "true" image
                        consecutive_false_count = 0
                        image_counter=0
                        break
                
                img_counter += 1
                start_time = time.time()

        if keyboard.is_pressed('alt+shift+d'):
            if not paused:
                notification.notify(
                    title="DND MODE STARTED",
                    message="Pausing camera. Press 'alt+shift+d' again to resume.",
                    timeout=5
                )
                paused = True
                show_notifications = False
                cam.release()
            else:
                notification.notify(
                    title="DND MODE ENDED",
                    message="Resuming Camera Capture",
                    timeout=5
                )
                notification.notify(
                    title="Take A Break",
                    message="Rest your eyes and look at a far away object",
                    #app_icon = r'D:\\coding\\Python\\break.ico',
                    timeout=user_input["number2"]
                )
                time.sleep(5+user_input["number2"])
                temp = 1
                paused = False
                show_notifications = True
                break
                cam = cv2.VideoCapture(0)
    
    cam.release()
    cv2.destroyAllWindows()
    # Your existing webcam monitoring code goes here
    # ...

# Create a thread for webcam monitoring
# webcam_thread = threading.Thread(target=webcam_monitoring)


def start_webcam_monitoring():
    webcam_thread = threading.Thread(target=webcam_monitoring)
    webcam_thread.start()
    return webcam_thread

# Main execution
while True:
    if __name__ == "__main__":
        webcam_thread = threading.Thread(target=webcam_monitoring)
        if temp == 0:
            gui_thread = threading.Thread(target=run_gui)
            gui_thread.start()
            gui_thread.join()  # Wait for the GUI thread to finish before starting webcam_thread
        webcam_thread.start()
    webcam_thread.join()