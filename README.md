# Emotion Detection from Video Stream

## Project Overview

This project focuses on developing a Convolutional Neural Network (CNN) model to classify emotions from facial expressions in a video stream. The model detects seven distinct emotions: Happy, Sad, Angry, Surprise, Fear, Disgust, and Neutral. The project includes both emotion classification and real-time emotion prediction from a webcam feed.


## Goals

	1.	Train a CNN to classify emotions based on images of faces, achieving over 60% accuracy on the test set.
	2.	Implement real-time emotion detection from a video stream, displaying detected emotions with confidence percentages.
	3.	Preprocess video frames to center faces and crop them for accurate emotion detection.

## Project Structure


```
project
├── data
│   ├── test_with_emotions.csv
│   ├── train.csv
├── requirements.txt
├── README.md
├── results
│   ├── model
│   │   ├── learning_curves.png
│   │   ├── final_emotion_model_arch.txt
│   │   ├── final_emotion_model.keras
│   │   ├── tensorboard.png
│   │   └── mnist_cnn_model.keras  # A foundational CNN model trained on MNIST to understand CNNs, aiding in training the emotion detection model.
│   └── preprocessing_test
│       ├── image0.png
│       ├── image_n.png
│       └── input_video.mp4
└── scripts
    ├── validation_loss_accuracy.py
    ├── predict_live_stream.py
    ├── predict.py
    ├── preprocess.py
    └── train.py
``` 

## Installation Instructions

1. Clone the repository:

    ``` 
    git clone https://01.kood.tech/git/dvorontso/emotions-detector
    ```

2. Create a virtual environment (recommended):

    ```
    python3.10 -m venv .venv
    source .venv/bin/activate   # On MacOS/Linux
    .venv\Scripts\activate      # On Windows
    ```


3. This project requires Python 3.10.12 for compatibility with TensorFlow. To install the necessary dependencies, run the following       command:

    ```
    pip install -r requirements.txt
    ```

 4. Ensure TensorBoard is installed to monitor model training.

    ```
    tensorboard --version
    pip install tensorboard
    ```

## Usage Instructions

1. Training the Model

    Run the training script with the following command:
    ```
    python scripts/train.py
    ```

    After training completes:
	•	Open TensorBoard to analyze the training results (new terminal):
    ```
    tensorboard --logdir=logs
    ```
    Go to http://localhost:6006 in your browser to view the training logs.
	•	Generate and save the learning curve plots:
    ```
    python scripts/validation_loss_accuracy.py
    ```

2. Real-Time Emotion Prediction

    To run real-time emotion prediction using the webcam feed, execute:
    ```
    python scripts/predict_live_stream.py
    ```
    The script will display detected emotions and their probabilities in real time.

3. Preprocessing the Video Stream

    To preprocess the video and save individual face images, execute:
    ```
    python scripts/preprocess.py
    ```
    This script saves faces from the video in the results/preprocessing_test folder at 48x48 resolution in grayscale.

4. Emotion Classification on Images

    To predict emotions on preprocessed images, run:

    ```
    python scripts/predict.py
    ```
    This script will display predicted emotions for each image in the results/preprocessing_test folder.

    Results Files

### The main results are stored in the results/model/ folder:
	•	final_emotion_model_arch.txt: Model architecture.
	•	learning_curves.png: Plot of training and validation accuracy and loss.
	•	tensorboard.png: Screenshot from TensorBoard showing the training progress.

### Training Data

The training data includes:

- **train.csv** and **test_with_emotions.csv**: Training and test datasets. These files should be placed in the `data/` folder in your project structure.

You can download them from the [Facial Expression Recognition Challenge on Kaggle](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data). Please note that a Kaggle account is required to access this dataset.

### Dependencies

The main dependencies for this project are:
	•	TensorFlow
	•	Keras
	•	OpenCV
	•	Matplotlib
	•	NumPy

See requirements.txt for the full list of dependencies.

### .gitignore Structure

The .gitignore file in this repository excludes large files and data results:
	•	Entire data/, .venv/, logs/, and results/preprocessing_test/ folders are ignored.
	•	Only essential files in results/model/ are saved.

