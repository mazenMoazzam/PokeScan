import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('fake_pokemon_card_detector.h5')

img_path = r"C:\Users\mazmo\OneDrive\Pictures\Screenshots\realll.jpg"  # Change extension as needed


def load_and_preprocess_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


image = load_and_preprocess_image(img_path)

prediction = model.predict(image)
predicted_class = (prediction > 0.5).astype("int32")[0][0]

if predicted_class == 1:
    print("The image is predicted to be Real.")
else:
    print("The image is predicted to be Fake.")
