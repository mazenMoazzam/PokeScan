import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('fake_pokemon_card_detector.h5')

# Path to the image you want to test
img_path = r"C:\Users\mazmo\OneDrive\Pictures\Screenshots\realll.jpg"  # Change extension as needed

# Function to load and preprocess image
def load_and_preprocess_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


image = load_and_preprocess_image(img_path)

prediction = model.predict(image)
predicted_class = (prediction > 0.5).astype("int32")[0][0]

if predicted_class == 1:
    print("The image is predicted to be Real.")
else:
    print("The image is predicted to be Fake.")
