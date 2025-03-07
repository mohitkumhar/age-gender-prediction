import cv2 as cv
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('model')  # Ensure your model is trained for age/gender detection

# Load Haar cascade for face detection
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

    # Convert frame to grayscale for face detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        face_resized = cv.resize(face, (200, 200))  # Resize to match model input size
        face_resized = face_resized / 255.0  # Normalize pixel values
        face_expanded = np.expand_dims(face_resized, axis=0)  # Add batch dimension

        # Make predictions
        age_pred, gender_pred = model.predict(face_expanded)

        # Process predictions
        predicted_age = int(age_pred[0][0])  # Assuming regression output
        gender_label = "Male" if gender_pred[0][0] < 0.5 else "Female"  # Assuming binary classification

        # Draw rectangle around the face
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add text above the rectangle
        text = f"{gender_label}, Age: {predicted_age}"
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

    cv.imshow('Age & Gender Detection', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
