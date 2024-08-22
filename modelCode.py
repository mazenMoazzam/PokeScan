

import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import regularizers

#Loading the csv files that contain the labels for the images of the pokemon cards.
train_labels = pd.read_csv(r"C:\Users\mazmo\Downloads\fakePokemonCardSet\train_labels.csv")
test_labels = pd.read_csv(r"C:\Users\mazmo\Downloads\fakePokemonCardSet\test_labels.csv")

#directories of the images of the cards.
train_dir = r"C:\Users\mazmo\Downloads\fakePokemonCardSet\train"
test_dir = r"C:\Users\mazmo\Downloads\fakePokemonCardSet\test"

#Added these lines to verify the columns were correct and the data was loaded correctly.
assert 'id' in train_labels.columns, "Column 'id' not found in train_labels.csv"
assert 'label' in train_labels.columns, "Column 'label' not found in train_labels.csv"
assert 'id' in test_labels.columns, "Column 'id' not found in test_labels.csv"
assert 'label' in test_labels.columns, "Column 'label' not found in test_labels.csv"


train_labels['file_path'] = train_labels['id'].apply(lambda x: os.path.join(train_dir, f"{x}.jpg"))
#used lambda to apply the full file path for each image and added to the dats frame.
test_labels['file_path'] = test_labels['id'].apply(lambda x: os.path.join(test_dir, f"{x}.jpg"))

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 15

#Preprocessing the images by resizing the images, reading it, and normalizing the pixel values for the images.
def loadAndPreprocessImage(filePath):
    img = cv2.imread(filePath)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img

trainImages = np.array([loadAndPreprocessImage(fp) for fp in train_labels['file_path']]) #applies the function
#above to all the images in the test set and training set and stores them in a numpy array.
testImages = np.array([loadAndPreprocessImage(fp) for fp in test_labels['file_path']])

trainLabelsArray = train_labels['label'].values
testLabelArray = test_labels['label'].values #extracts labels as numpy array

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(trainImages, trainLabelsArray, test_size=0.2, random_state=42)
#splits the training images and labels into a training and validation set.

#Overall CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#complied model with adaptive moment estimation optimizer. Chose this as it improves training speeds in the neural network
#and will reach convergence at a quicker rate.

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

# Evaluate model
test_loss, test_acc = model.evaluate(testImages, testLabelArray)
print(f'Test accuracy: {test_acc:.4f}')

# Predict and print classification report
y_pred = (model.predict(testImages) > 0.5).astype("int32")
print(classification_report(testLabelArray, y_pred, target_names=['Fake', 'Real']))