from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import io
import pickle

app = Flask(__name__)

# Load the model from pickle file (assuming you have this saved already)
music_model = pickle.load(open('model.pkl', 'rb'))

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Retrieve the uploaded audio file
            file = request.files['music']
            
            # Check if the file is a WAV file
            if not file.filename.endswith('.wav'):
                return render_template('index.html', result="Please upload a valid WAV file.")
            
            # Read the file as a byte stream and decode it
            audio_file = file.read()
            audio_tensor = tf.io.decode_wav(audio_file, desired_channels=1, desired_samples=16000)
            waveform = audio_tensor.audio[0]  # Extract the waveform (audio signal)

            # Convert the waveform to a spectrogram
            spectrogram = get_spectrogram(waveform)

            # Use the model to predict
            prediction = music_model(spectrogram[tf.newaxis, ...])

            # Get predicted class
            predicted_class = tf.argmax(prediction, axis=-1).numpy()[0]
            predicted_label = label_names[predicted_class]

            # Render the result on the home page
            return render_template('index.html', result=f'The prediction is: {predicted_label}')
        except Exception as e:
            return render_template('index.html', result=f"left")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
