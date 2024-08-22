import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras import regularizers

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

IMG_SIZE = (128, 128)
BATCH_SIZE = 15
EPOCHS = 10
k = 5  #number of folds that I will run


def loadAndPreprocessImage(filePath):
    img = cv2.imread(filePath)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img


trainImages = np.array([loadAndPreprocessImage(fp) for fp in train_labels['file_path']])
testImages = np.array([loadAndPreprocessImage(fp) for fp in test_labels['file_path']])

trainLabelsArray = train_labels['label'].values
testLabelArray = test_labels['label'].values

kf = KFold(n_splits=k, shuffle=True, random_state=42)
fold_no = 1
acc_scores = []

for train_index, val_index in kf.split(trainImages):
    print(f'Training on fold {fold_no}...')

    X_train, X_val = trainImages[train_index], trainImages[val_index]
    y_train, y_val = trainLabelsArray[train_index], trainLabelsArray[val_index]

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01),
                      input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
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

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

    val_loss, val_acc = model.evaluate(X_val, y_val)
    acc_scores.append(val_acc)
    print(f'Fold {fold_no} Validation accuracy: {val_acc:.4f}')

    fold_no += 1

print(f'Average validation accuracy across {k} folds: {np.mean(acc_scores):.4f}')

test_loss, test_acc = model.evaluate(testImages, testLabelArray)
print(f'Test accuracy: {test_acc:.4f}')

y_pred = (model.predict(testImages) > 0.5).astype("int32")
print(classification_report(testLabelArray, y_pred, target_names=['Fake', 'Real']))


