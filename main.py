import cv2

# Load the Haar cascade files for face and eye detection
def load_cascades():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

# Capture video frame by frame
def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def detect_faces(face_cascade, gray_frame, scale_factor = 1.1, minNeigbors = 4):
    faces = face_cascade.detectMultiScale(gray_frame, scale_factor, minNeigbors)
    return faces

def compute_rectangle(faces, frame, draw = True, color = (255, 0, 0), thickness = 2):
    corners = []
    for (x, y, w, h) in faces:
        corner_1 = (x, y)
        corner_2 = (x + w, y + h)
        corners.append([corner_1, corner_2])
        if draw:
            cv2.rectangle(frame, corner_1, corner_2, color, thickness)

    return corners

def compute_midpoint(frame, rectangels, draw = True, radius = 5, color = (0, 0, 255), thickness = -1):
    midpoints = []
    for rect in rectangels:
        x = (rect[0][0] + rect[1][0]) // 2
        y = (rect[0][1] + rect[1][1]) // 2
        midpoints.append((x, y))
        if draw:
            cv2.circle(frame, (x,y), radius, color, thickness)  # Red dot for eye center
    
    return midpoints

def compute_error(image_center, face_midpoints, frame, draw = True):
    errors = []
    for face_center in face_midpoints:
        ex = image_center[0] - face_center[0]
        ey = image_center[1] - face_center[1]
        errors.append((ex, ey))
        if draw:
            print((ex, ey), image_center, face_center)
            cv2.line(frame, face_center, (ex+face_center[0], ey+face_center[1]), (255, 255, 255), 2)
    return errors


# Process and display the video stream
def process_video_stream(cap, face_cascade):
    image_center = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
        )

    
    while True:
        # Capture frame-by-frame
        frame = capture_frame(cap)
        if frame is None:
            break

        # Convert the frame to grayscale for detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detect_faces(face_cascade, gray_frame, 1.1, 7)
        rects = compute_rectangle(faces, frame, False)
        face_midpoints = compute_midpoint(frame, rects, color=(0,255,0))
        error = compute_error(image_center, face_midpoints, frame)
        cv2.circle(frame, image_center, 5, (0,0,255), -1)  # Red dot for eye center
 
        
        # Display the resulting frame
        flipped_frame = cv2.flip(frame, 1)
        cv2.imshow('Face and Eye Tracking', flipped_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Main function to start the program
def main():
    # Load the cascades for face and eye detection
    face_cascade = load_cascades()

    # Open the external webcam (change the index to 1, 2, etc., if needed)
    cap = cv2.VideoCapture(0)  # Use 2 for the external webcam and 0 for build-in camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Process the video stream
    process_video_stream(cap, face_cascade)

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    main()
