
import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
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
N_FOLDS = 5


def loadAndPreprocessImage(filePath):
    img = cv2.imread(filePath)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img


trainImages = np.array([loadAndPreprocessImage(fp) for fp in train_labels['file_path']])
trainLabelsArray = train_labels['label'].values

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_accuracies = []

for fold, (train_index, val_index) in enumerate(kf.split(trainImages)):
    print(f"\nFold {fold + 1}/{N_FOLDS}")

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
    fold_accuracies.append(val_acc)
    print(f'Validation accuracy for fold {fold + 1}: {val_acc:.4f}')

print(f'\nAverage validation accuracy across {N_FOLDS} folds: {np.mean(fold_accuracies):.4f}')

testImages = np.array([loadAndPreprocessImage(fp) for fp in test_labels['file_path']])
testLabelArray = test_labels['label'].values
test_loss, test_acc = model.evaluate(testImages, testLabelArray)
print(f'\nTest accuracy: {test_acc:.4f}')


