import cv2
import numpy as np
import tensorflow as tf
import os

# Loading the trained model
model = tf.keras.models.load_model("../results/model/final_emotion_model.keras")

# Setting up a Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Path to the recorded video
video_path = "../results/preprocessing_test/input_video.mp4"
output_folder = "../results/preprocessing_test"

#Create a folder to save frames if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Opening video
cap = cv2.VideoCapture(video_path)
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second

# Dictionary for displaying emotions
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

frame_interval = frame_rate  # Interval to get 1 frame per second
saved_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Process each frame at 1 second intervals
    if frame_index % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        if len(faces) > 0:
            # If a face is found, we take the first one and predict the emotion
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            face_normalized = face_resized / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

            # Predicting emotion
            prediction = model.predict(face_reshaped)
            emotion_index = np.argmax(prediction)
            emotion_label = emotion_labels[emotion_index]
            confidence = np.max(prediction) * 100

           # Output predicted emotion and probability
            print(f"Эмоция: {emotion_label}, вероятность: {confidence:.2f}%")

            # Save frame
            frame_filename = os.path.join(output_folder, f"image{saved_frame_count}.png")
            cv2.imwrite(frame_filename, face_resized)
            print(f"Сохранен кадр: {frame_filename} для эмоции {emotion_label} с вероятностью {confidence:.2f}%")
        else:
            print(f"Лицо не найдено на кадре {frame_index}")

        saved_frame_count += 1

# Finishing working with video
cap.release()
cv2.destroyAllWindows()
print("Обработка и сохранение кадров завершены.")