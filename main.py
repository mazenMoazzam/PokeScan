import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

train_labels = pd.read_csv(r"C:\Users\mazmo\Downloads\fakePokemonCardSet\train_labels.csv")
test_labels = pd.read_csv(r"C:\Users\mazmo\Downloads\fakePokemonCardSet\test_labels.csv")

train_dir = r"C:\Users\mazmo\Downloads\fakePokemonCardSet\train"
test_dir = r"C:\Users\mazmo\Downloads\fakePokemonCardSet\test"

assert 'id' in train_labels.columns, "Column 'id' not found in train_labels.csv"
assert 'label' in train_labels.columns, "Column 'label' not found in train_labels.csv"
assert 'id' in test_labels.columns, "Column 'id' not found in test_labels.csv"
assert 'label' in test_labels.columns, "Column 'label' not found in test_labels.csv"

train_labels['file_path'] = train_labels['id'].apply(lambda x: os.path.join(train_dir, f"{x}.jpg"))
test_labels['file_path'] = test_labels['id'].apply(lambda x: os.path.join(test_dir, f"{x}.jpg"))

IMG_SIZE = (128,128)
BATCH_SIZE = 32

def loadAndPreprocessImage(filePath):
    img = cv2.imread(filePath)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img

trainImages = np.array([loadAndPreprocessImage(fp) for fp in train_labels['file_path']])
testImages = np.array([loadAndPreprocessImage(fp) for fp in test_labels['file_path']])

trainLabelsArray = train_labels['label'].values
testLabelArray = test_labels['label'].values

X_train, X_val, y_train, y_val = train_test_split(trainImages, trainLabelsArray, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') #used sigmoid activiation function as ending result will be a binary number.
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

test_loss, test_acc = model.evaluate(testImages, testLabelArray)
print(f'Test accuracy: {test_acc:.4f}')

y_pred = (model.predict(testImages) > 0.5).astype("int32")
print(classification_report(testLabelArray, y_pred, target_names=['Fake', 'Real']))
