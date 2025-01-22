import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np

# Reading data
train_data = pd.read_csv("../data/train.csv", sep=",", header=0)
test_data = pd.read_csv("../data/test_with_emotions.csv", sep=",", skiprows=1, index_col=0, names=["emotion", "pixels"])

# Converting pixels to an array of images and labels
x_train = np.array([np.fromstring(pixels, dtype=int, sep=' ').reshape(48, 48, 1) for pixels in train_data['pixels']])
y_train = train_data['emotion'].values
x_test = np.array([np.fromstring(pixels, dtype=int, sep=' ').reshape(48, 48, 1) for pixels in test_data['pixels']])
y_test = test_data['emotion'].values

# Data normalization
x_train, x_test = x_train / 255.0, x_test / 255.0

# model CNN
model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
tensorboard_callback = TensorBoard(log_dir="../logs", histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True
)
datagen.fit(x_train)

# Model training
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback, early_stopping]
)

# Model evaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Saving the model and architecture
model.save("../results/model/final_emotion_model.keras")
with open("../results/model/final_emotion_model_arch.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# Saving learning history for graphs
np.save('../results/model/training_history.npy', history.history)