import cv2
import numpy as np
import tensorflow as tf
import os

# Loading the trained model
model = tf.keras.models.load_model("../results/model/final_emotion_model.keras")

# Path to the folder with saved images
input_folder = "../results/preprocessing_test"

# Dictionary for displaying emotions
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Checking for the presence of the image folder
if not os.path.exists(input_folder):
    print("Ошибка: папка с изображениями не найдена.")
else:
    # Loop through all images in a folder
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".png"):
            # Path to image
            image_path = os.path.join(input_folder, filename)

            # Loading and preprocessing the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Ошибка: не удалось загрузить изображение {filename}")
                continue

           # Resize and normalize an image
            image_resized = cv2.resize(image, (48, 48)) / 255.0
            image_reshaped = np.reshape(image_resized, (1, 48, 48, 1))

           # Predicting emotion
            prediction = model.predict(image_reshaped)
            emotion_index = np.argmax(prediction)
            emotion_label = emotion_labels[emotion_index]
            confidence = np.max(prediction) * 100

            # Output predicted emotion and probability
            print(f"Файл: {filename} | Эмоция: {emotion_label} | Вероятность: {confidence:.2f}%")
            # Additional output of probabilities of all emotions for detail
            for idx, prob in enumerate(prediction[0]):
                print(f"  {emotion_labels[idx]}: {prob * 100:.2f}%")
            print("\n" + "-"*50 + "\n")