from flask import Flask, jsonify, request
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
app = Flask(__name__)

# Dữ liệu người dùng giả định
users = [
    {"id": 1, "name": "John"},
    {"id": 2, "name": "Alice"},
    {"id": 3, "name": "Bob"}
]

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify(users)

@app.route("/api/receive-data",methods = ['POST'])
def receiveData():
    data = request.get_json()
    print(data)
    result = int(processData(data))
    return jsonify(result)
    
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3
model_path = 'assign_model.h5'
image_path = 'test.png'
def processData(data):
    dataList = []
    model = keras.models.load_model(model_path)
    image = cv2.imread(image_path)
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
    dataList.append(np.array(resize_image))
    X_test = np.array(dataList)
    X_test = X_test/255
    pred = model.predict(X_test)
    print('-------------')
    array = pred[0]
    print(len(array))
    print(type(array))
    max_index = np.argmax(array)
    print(max_index)
    print('-------------')
    return max_index

@app.route("/api/upload", methods=["POST"])
def upload():
    image = request.files["image"]
    name = request.form.get("name")
    name = "name "+str(name)
    print(name)
    # Lưu ảnh vào thư mục "images"
    cv2.imwrite("demo.png", image)
    dataList = []
    model = keras.models.load_model(model_path)
    image = cv2.imread(image)
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
    dataList.append(np.array(resize_image))
    X_test = np.array(dataList)
    X_test = X_test/255
    pred = model.predict(X_test)
    print('-------------')
    array = pred[0]
    print(len(array))
    print(type(array))
    max_index = np.argmax(array)
    print(max_index)
    print('-------------')
    # Trả về thông báo thành công
    return "Image uploaded successfully\nId: " + str(max_index)