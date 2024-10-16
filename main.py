import cv2

# Load the Haar cascade files for face and eye detection
def load_cascades():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cascade, eye_cascade

# Capture video frame by frame
def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

# Detect faces in a frame
def detect_faces(face_cascade, gray_frame):
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
    return faces

# Detect eyes within the region of detected faces
def detect_eyes(eye_cascade, face_roi_gray):
    eyes = eye_cascade.detectMultiScale(face_roi_gray)
    return eyes

def eyes_center(frame, faces, gray_frame, eye_cascade):
    eyes_center = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Region of interest (ROI) for eyes within the face
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = detect_eyes(eye_cascade, roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_center = (ex + ew // 2, ey + eh // 2)
            eyes_center.append(eye_center)
            cv2.circle(roi_color, eye_center, 5, (0, 0, 255), -1)  # Red dot for eye center
        
    return eyes_center

def eyes_midpoint(frame, eyes_center):
    midpoint = []
    if len(eyes_center) > 1:

        midpoint = [
            (eyes_center[0][0] + eyes_center[1][0]) // 2, 
            (eyes_center[0][1] + eyes_center[1][1]) // 2
            ]
    else:
        if len(eyes_center) == 0:
            midpoint = [-1, -1]
        else:
            midpoint = [eyes_center[0][0], eyes_center[0][1]]

    cv2.circle(frame , midpoint, 5, (0, 255, 0), -1)  # Red dot for eye center
    

# Process and display the video stream
def process_video_stream(cap, face_cascade, eye_cascade):
    while True:
        # Capture frame-by-frame
        frame = capture_frame(cap)
        if frame is None:
            break

        # Convert the frame to grayscale for detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detect_faces(face_cascade, gray_frame)

        # Draw rectangles around faces and circles at eye centers
        eyes_position = eyes_center(frame, faces, gray_frame, eye_cascade)
        eyes_midpoint(frame, eyes_position)
        
        
        # Display the resulting frame
        cv2.imshow('Face and Eye Tracking', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Main function to start the program
def main():
    # Load the cascades for face and eye detection
    face_cascade, eye_cascade = load_cascades()

    # Open the external webcam (change the index to 1, 2, etc., if needed)
    cap = cv2.VideoCapture(2)  # Use 2 for the external webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Process the video stream
    process_video_stream(cap, face_cascade, eye_cascade)

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    main()
