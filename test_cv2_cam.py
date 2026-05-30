import cv2
import time

print("Attempting to initialize native camera with DirectShow...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Success: Webcam opened successfully.")
    # Warm up camera
    ret = False
    for i in range(15):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i} read successfully!")
        time.sleep(0.1)
    if ret:
        print(f"Captured frame shape: {frame.shape}")
        cv2.imwrite("captured_test.jpg", frame)
        print("Saved test frame to captured_test.jpg")
    else:
        print("Error: Failed to read frame.")
    cap.release()
