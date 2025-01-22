import cv2
import numpy as np
import tensorflow as tf
import os
import time

# Loading the trained model
model = tf.keras.models.load_model("../results/model/final_emotion_model.keras")

# Setting up a Haar Cascade for Face Detection (Pre-trained model in OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Setting up a folder for recording video
output_folder = "../results/preprocessing_test"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Setting up a webcam to capture video
cap = cv2.VideoCapture(0)  # Index 0 usually corresponds to the main webcam
ret, frame = cap.read()

# Checking if frame capture was successful
if not ret:
    print("Ошибка: не удалось получить кадр с веб-камеры.")
    cap.release()
else:
    frame_height, frame_width = frame.shape[:2]  # Determine the frame dimensions

 # Setting up video recording by determining resolution from frames
    video_path = os.path.join(output_folder, "input_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (frame_width, frame_height))

if not out.isOpened():
    print("Ошибка: не удалось открыть файл для записи видео.")
else:
    print("Запись видео начата.")

# Vocabulary for displaying emotions
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

start_time = time.time()

# Record a video for 20 seconds
while time.time() - start_time <= 20:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось получить кадр с веб-камеры.")
        break

    
# Converting a frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection in a frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    for (x, y, w, h) in faces:
       # Face Area Extraction
        face = gray[y:y+h, x:x+w]

        # Pre-processing the face for the model
        face_resized = cv2.resize(face, (48, 48))  # Resizing to 48x48
        face_normalized = face_resized / 255.0  # Normalization
        face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))  # Changing the shape for the model

        # Predicting emotion
        prediction = model.predict(face_reshaped)
        emotion_index = np.argmax(prediction)  # Maximum Probability Index
        emotion_label = emotion_labels[emotion_index]
        confidence = np.max(prediction) * 100  # Probability in percent

        # Display predicted emotion and probability on frame
        label = f"{emotion_label} ({confidence:.2f}%)"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw a rectangle around the face

    # Write frame to file
    out.write(frame)
    cv2.imshow("Emotion Detection", frame)

    # Exit the program by pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Запись видео завершена.")
        break

# Resource release
cap.release()
out.release()  #Close video file
cv2.destroyAllWindows()
print(f"Время записи: {time.time() - start_time:.2f} секунд")
