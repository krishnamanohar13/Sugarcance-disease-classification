from flask import Flask, render_template, request
import numpy as np
import tensorflow
import keras 
from keras import load_model  
from PIL import Image
from utils import img_to_array
from skimage.color import rgb2yuv,yuv2rgb
from scipy.signal import convolve2d
import os

UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("Final_DenseNet201_Model.h5")
sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

def multi_convolver(image, kernel, iterations):
    for i in range(iterations):
        image = convolve2d(image, kernel, 'same', boundary = 'fill',
                           fillvalue = 0)
    return image

def convolver_rgb(image, kernel, iterations = 1):
    img_yuv = rgb2yuv(image)   
    img_yuv[:,:,0] = multi_convolver(img_yuv[:,:,0], kernel, 
                                     iterations)
    final_image = yuv2rgb(img_yuv)
    return final_image

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        if file is None:
            return "No file uploaded"
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        img = Image.open(file)
        img = img.resize((224, 224))
        x = img_to_array(img)
        x = x/255
        img = convolver_rgb(x, sharpen, iterations=1)
        x = x.reshape(224, 224, 3)
        x = np.expand_dims(x, axis=0)
        predi = model.predict(x)
        print(predi)
        classes_x = np.argmax(predi)
        print(classes_x)

        classes = ["Healthy", "Red Rot", "Red Rust"]
        prediction_label = classes[classes_x]

        print("Disease detected: " + prediction_label)
        output = prediction_label
        return render_template('output.html', output=output)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 
