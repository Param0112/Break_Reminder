import cv2
from deepface import DeepFace

# Load the pre-trained model for face recognition
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