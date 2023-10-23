from flask import Flask, jsonify, request
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image
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
    result = processData(data)
    return jsonify(result)
    

def processData(data):
    model = keras.models.load_model('assign_model.h5')
    image = np.array(Image.open("test.png"))
    resized_image = np.reshape(image, (-1, 30, 30, 3))
    predictions = model.predict(resized_image)
    print(predictions)
    return predictions