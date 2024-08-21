from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('model.h5')

img_path = r"C:\Users\mazmo\Downloads\carddd.jpg"

img = image.load_img(img_path, target_size=(128, 128))
imgArray = image.img_to_array(img) / 255.0
imgArray = np.expand_dims(imgArray, axis=0)

prediction = model.predict(imgArray)
predictedClass = np.argmax(prediction)

print(f'Predicted class: {predictedClass}')
