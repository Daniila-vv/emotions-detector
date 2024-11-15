import cv2
import numpy as np
import tensorflow as tf
import os

# Загрузка обученной модели
model = tf.keras.models.load_model("../results/model/final_emotion_model.keras")

# Настройка каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Путь к записанному видео
video_path = "../results/preprocessing_test/input_video.mp4"
output_folder = "../results/preprocessing_test"

# Создание папки для сохранения кадров, если её нет
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Открытие видео
cap = cv2.VideoCapture(video_path)
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Число кадров в секунду

# Словарь для отображения эмоций
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

frame_interval = frame_rate  # Интервал для получения 1 кадра в секунду
saved_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Обработка каждого кадра с интервалом в 1 секунду
    if frame_index % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        if len(faces) > 0:
            # Если лицо найдено, берем первое и предсказываем эмоцию
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            face_normalized = face_resized / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

            # Предсказание эмоции
            prediction = model.predict(face_reshaped)
            emotion_index = np.argmax(prediction)
            emotion_label = emotion_labels[emotion_index]
            confidence = np.max(prediction) * 100

            # Вывод предсказанной эмоции и вероятности
            print(f"Эмоция: {emotion_label}, вероятность: {confidence:.2f}%")

            # Сохранение кадра
            frame_filename = os.path.join(output_folder, f"image{saved_frame_count}.png")
            cv2.imwrite(frame_filename, face_resized)
            print(f"Сохранен кадр: {frame_filename} для эмоции {emotion_label} с вероятностью {confidence:.2f}%")
        else:
            print(f"Лицо не найдено на кадре {frame_index}")

        saved_frame_count += 1

# Завершение работы с видео
cap.release()
cv2.destroyAllWindows()
print("Обработка и сохранение кадров завершены.")