import cv2
import face_recognition
import face_recognition_models

# Load the Haar cascade files for face detection
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
            cv2.circle(frame, (x, y), radius, color, thickness)  # Red dot for face center
    
    return midpoints

def compute_error(image_center, face_midpoints, frame, draw = True):
    errors = []
    for face_center in face_midpoints:
        ex = image_center[0] - face_center[0]
        ey = image_center[1] - face_center[1]
        errors.append((ex, ey))
        if draw:
            print((ex, ey), image_center, face_center)
            cv2.line(frame, face_center, (ex + face_center[0], ey + face_center[1]), (255, 255, 255), 2)
    return errors

# Process and display the video stream
def process_video_stream(cap, face_cascade, known_face_encodings, known_face_names):
    image_center = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
    )

    while True:
        # Capture frame-by-frame
        frame = capture_frame(cap)
        if frame is None:
            break

        # Convert the frame to grayscale for face detection (Haar cascades)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascade
        faces = detect_faces(face_cascade, gray_frame, 1.1, 7)
        rects = compute_rectangle(faces, frame, False)
        face_midpoints = compute_midpoint(frame, rects, color=(0, 255, 0))
        error = compute_error(image_center, face_midpoints, frame)

        # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
        rgb_frame = frame[:, :, ::-1]

        # Use face_recognition to find all face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop over each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face is a match for any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"  # Default to Unknown if no match is found

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw the label (name) below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # Display the resulting frame
        flipped_frame = cv2.flip(frame, 1)
        cv2.imshow('Face Tracking and Recognition', flipped_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Main function to start the program
def main():
    # Load the Haar Cascade for face detection
    face_cascade = load_cascades()

    # Load known face images and encode them
    # Replace these image paths with actual paths to your known face images
    known_face_encodings = []
    known_face_names = []

    person1_image = face_recognition.load_image_file("person1.jpeg")
    person2_image = face_recognition.load_image_file("person2.jpeg")

    known_face_encodings.append(face_recognition.face_encodings(person1_image)[0])
    known_face_encodings.append(face_recognition.face_encodings(person2_image)[0])

    known_face_names.append("Niki")
    known_face_names.append("JV")

    # Open the webcam (change the index to 1, 2, etc., if needed)
    cap = cv2.VideoCapture(0)  # Use 2 for an external webcam and 0 for the built-in camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Process the video stream with face recognition
    process_video_stream(cap, face_cascade, known_face_encodings, known_face_names)

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    main()
