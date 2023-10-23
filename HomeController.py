from flask import Flask, jsonify, request
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
app = Flask(__name__)

# Dữ liệu người dùng giả định
users = [
    {"id": 1, "name": "John"},
    {"id": 2, "name": "Alice"},
    {"id": 3, "name": "Bob"}
]

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