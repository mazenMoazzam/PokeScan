import numpy as np
from PIL import Image
import io
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('fake_pokemon_card_detector.h5')


@app.route('/model_info', methods=['GET'])
def model_info():
    try:
        modelSummary = []
        model.summary(print_fn=lambda x: modelSummary.append(x))
        return jsonify({'Model Summary': '\n'.join(modelSummary)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            image = image.resize((128, 128))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            predictionOutcome = model.predict(image)
            label = np.argmax(predictionOutcome, axis=1)[0]
            return jsonify({'Prediction': int(label)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

