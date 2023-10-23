from flask import Flask, jsonify, request
from keras.models import load_model
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
    # try:
    model = load_model('resources\assign_model.h5')
    image = np.array(Image.open("resources\img\test.png"))
    predictions = model.predict(image)
    print(predictions)
    return predictions
    # except:
    #     return ""