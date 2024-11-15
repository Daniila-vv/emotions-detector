import cv2
import numpy as np
import tensorflow as tf
import os
import time

# Загрузка обученной модели
model = tf.keras.models.load_model("../results/model/final_emotion_model.keras")

# Настройка каскада Хаара для обнаружения лиц (предобученная модель в OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Настройка папуи для записи видео
output_folder = "../results/preprocessing_test"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Настройка веб-камеры для захвата видео
cap = cv2.VideoCapture(0)  # Индекс 0 обычно соответствует основной веб-камере
ret, frame = cap.read()

# Проверка успешности захвата кадра
if not ret:
    print("Ошибка: не удалось получить кадр с веб-камеры.")
    cap.release()
else:
    frame_height, frame_width = frame.shape[:2]  # Определяем размеры кадра

 # Настройка записи видео с определением разрешения из кадров
    video_path = os.path.join(output_folder, "input_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Кодек для MP4
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (frame_width, frame_height))

if not out.isOpened():
    print("Ошибка: не удалось открыть файл для записи видео.")
else:
    print("Запись видео начата.")

# Словарь для отображения эмоций
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

start_time = time.time()

# Записываем видео 20 секунд
while time.time() - start_time <= 20:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось получить кадр с веб-камеры.")
        break

    
# Преобразование кадра в градации серого для обнаружения лиц
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    for (x, y, w, h) in faces:
        # Извлечение области лица
        face = gray[y:y+h, x:x+w]

        # Шаг 5: Предобработка лица для модели
        face_resized = cv2.resize(face, (48, 48))  # Изменение размера на 48x48
        face_normalized = face_resized / 255.0  # Нормализация
        face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))  # Изменение формы для модели

        # Предсказание эмоции
        prediction = model.predict(face_reshaped)
        emotion_index = np.argmax(prediction)  # Индекс с максимальной вероятностью
        emotion_label = emotion_labels[emotion_index]
        confidence = np.max(prediction) * 100  # Вероятность в процентах

        # Отображение предсказанной эмоции и вероятности на кадре
        label = f"{emotion_label} ({confidence:.2f}%)"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Рисуем прямоугольник вокруг лица

    # Запись кадра в файл
    out.write(frame)
    cv2.imshow("Emotion Detection", frame)

    # Выход из программы по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Запись видео завершена.")
        break

# Освобождение ресурсов
cap.release()
out.release()  # Закрытие видеофайла
cv2.destroyAllWindows()
print(f"Время записи: {time.time() - start_time:.2f} секунд")
