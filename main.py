import cv2 as cv

# Load Haar cascade face detector
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error loading face cascade. Check OpenCV installation.")
    exit()

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Camera opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add text above the rectangle
        text = "Face"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 255, 0)  # Green color
        thickness = 2
        text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
        
        # Calculate text position (above the face)
        text_x = x + (w - text_size[0]) // 2
        text_y = y - 10 if y - 10 > 10 else y + 20  # Prevent text from going off-screen

        # Put text on the frame
        cv.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)

    cv.imshow('Face Detection', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
