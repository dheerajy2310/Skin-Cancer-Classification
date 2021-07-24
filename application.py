import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from skimage.transform import resize
from flask import Flask, url_for, request, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/images/'

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


weights_path="static/models/weights.h5"

# model=tf.keras.models.load_model(model_path)
resnet_model=ResNet50(include_top=False,weights=None,input_shape=(224,224,3))
for i in range(175):
    resnet_model.layers[i].trainable=False
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (224,224))))
model.add(resnet_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256,activation='relu',))
model.add(tf.keras.layers.Dense(128,activation='relu',))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64,activation='relu',))
model.add(tf.keras.layers.Dense(7,activation='softmax',))
model.build(input_shape=(None,224,224,3))

model.load_weights(weights_path)
classes=["actinic keratosis","basal cell carcinoma","benign lesions of the keratosis type","dermatofibroma","melanoma","nevus","vascular lesion"]
def model_predict(img_path,model):
    img=load_img(img_path,target_size=(224,224))
    img=img_to_array(img)
    img=resize(img,(224,224))
    img=np.reshape(img,(1,)+(224,224,3))
    return classes[np.argmax(model.predict(img))]

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])
def predict():
    f=request.files['file']
    basepath=os.path.dirname(__file__)
    file_path=os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
    f.save(file_path)
    preds=model_predict(file_path,model)
    return preds

if __name__ == '__main__':
    app.run(debug=True)
    
