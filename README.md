# Sleep Detection CNN Model
This repository contains code for a Convolutional Neural Network (CNN) classification model designed to predict whether a person is sleeping or not based on the detection of closed eyes. The model also takes into account yawning as an additional feature for sleep detection.

# Dataset
The model has been trained on a dataset consisting of images with labels indicating whether the person in the image is sleeping or not. The dataset includes variations in lighting conditions, head poses, and facial expressions.
# Model Architecture
The CNN model architecture is as follows:
model = Sequential()
# code
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# end of code
This architecture consists of convolutional layers for feature extraction, max-pooling layers for down-sampling, and fully connected layers for classification. The final layer uses the sigmoid activation function for binary classification.

# Model Training
The dataset was split into training, validation, and test sets with the following proportions: 70% training, 20% validation, and 10% test. The model was trained using binary cross-entropy loss and the Adam optimizer.
# data set used
https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset?select=train 
