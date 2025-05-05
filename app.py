from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Modelleri yükle
#cnn_model = load_model('model/cnn_model.h5')        # CNN tabanlı modelin yolu
#vgg16_model = load_model('model/vgg16_model.h5')    # VGG16 tabanlı modelin yolu

# Ana Sayfa
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Görseli yükle ve işleme al
            img = image.load_img(filepath, target_size=(224, 224))  # Gerekirse boyut modeline göre değiştir
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalizasyon

            # CNN Model Tahmini
            cnn_pred = cnn_model.predict(img_array)
            cnn_result = 'Zatürre' if cnn_pred[0][0] > 0.5 else 'Normal'

            # VGG16 Model Tahmini
            vgg16_pred = vgg16_model.predict(img_array)
            vgg16_result = 'Zatürre' if vgg16_pred[0][0] > 0.5 else 'Normal'

            # Sonuçları gönder
            return render_template('index.html', cnn_result=cnn_result, vgg16_result=vgg16_result)

    return render_template('index.html', cnn_result=None, vgg16_result=None)

# Uygulamayı çalıştır
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
