import cv2
import numpy as np
import tensorflow as tf
import os

# Загрузка обученной модели
model = tf.keras.models.load_model("../results/model/final_emotion_model.keras")

# Путь к папке с сохраненными изображениями
input_folder = "../results/preprocessing_test"

# Словарь для отображения эмоций
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Проверка наличия папки с изображениями
if not os.path.exists(input_folder):
    print("Ошибка: папка с изображениями не найдена.")
else:
    # Перебор всех изображений в папке
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".png"):
            # Путь к изображению
            image_path = os.path.join(input_folder, filename)

            # Загрузка и предобработка изображения
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Ошибка: не удалось загрузить изображение {filename}")
                continue

            # Изменение размера и нормализация изображения
            image_resized = cv2.resize(image, (48, 48)) / 255.0
            image_reshaped = np.reshape(image_resized, (1, 48, 48, 1))

            # Предсказание эмоции
            prediction = model.predict(image_reshaped)
            emotion_index = np.argmax(prediction)
            emotion_label = emotion_labels[emotion_index]
            confidence = np.max(prediction) * 100

            # Вывод предсказанной эмоции и вероятности
            print(f"Файл: {filename} | Эмоция: {emotion_label} | Вероятность: {confidence:.2f}%")
            # Дополнительный вывод вероятностей всех эмоций для подробности
            for idx, prob in enumerate(prediction[0]):
                print(f"  {emotion_labels[idx]}: {prob * 100:.2f}%")
            print("\n" + "-"*50 + "\n")