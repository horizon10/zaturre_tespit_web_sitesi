from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model yüklemesi
#model = load_model('model/model.h5')  # kendi model yoluna göre değiştir

# Ana Sayfa
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Dosya yüklenirse
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Model tahmini
            img = image.load_img(filepath, target_size=(224, 224))  # kendi modeline göre boyut değişebilir
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # normalize

            prediction = model.predict(img_array)
            result = 'Zatürre' if prediction[0][0] > 0.5 else 'Normal'

            # Tahmin sonucunu göster
            return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
