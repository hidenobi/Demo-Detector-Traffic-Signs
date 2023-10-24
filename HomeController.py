from flask import Flask, jsonify, request
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from entities.LabelSys import LabelSys
from entities.BoxImage import BoxImage
from entities.AnswerSys import AnswerSys
from entities.ResultSys import ResultSys
from entities.ImageSys import ImageSys
import datetime
import json
import mysql.connector
app = Flask(__name__)
# cấu hình CDSL

mydb = mysql.connector.connect(
  host="localhost",
  user="sa",
  password="12345678",
  database="detectortrafficsigns"
)
mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM label_sys")

myresult = mycursor.fetchall()
classes = []
for i in myresult:
    l = LabelSys(id=i[0],name=str(i[2]),dateEdit=i[1])
    classes.append(l)

IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3
model_path = 'assign_model.h5'
image_path = 'demo.png'

detector_model_path = "detector_object.pt"
detector_model = YOLO(detector_model_path)

# main function for demo
assign_model = keras.models.load_model(model_path)
@app.route('/receive-data',methods = ["POST"])
def receiveData():
    image = request.files["image"]
    id_result = int(request.form.get("id"))
    # tìm vị trí vật thể
    image.save(image_path)
    results = detector_model(image_path)
    results = results[0]
    position = results.boxes.xyxy.tolist()
    (h,w) = results.orig_shape
    top = 0
    left = 0
    bottom = h
    right = w
    if results.boxes.id != None:
        top = position[0]
        left = position[1]
        bottom = position[2]
        right = position[3]
    boxImage = BoxImage(id=-1,top=top,left=left,bottom=bottom,right=right)
    # tiền xử lý ảnh cho assign model
    image = cv2.imread(image_path)
    # Cắt ảnh theo tọa độ của bốn điểm
    cropped_image = image[top:bottom,left:right]

    # tìm nhãn của vật thể
    dataList = []
    image = cropped_image
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
    dataList.append(np.array(resize_image))
    X_test = np.array(dataList)
    X_test = X_test/255
    pred = assign_model.predict(X_test)
    array = pred[0]
    max_index = np.argmax(array)
    index_label = int(max_index)
    labelSys = classes[index_label]

    # xử lý kết quả
    imageSys = ImageSys(id=id_result,path=image_path) 
    answerSys = AnswerSys(id=id_result,boxImage=boxImage,labelSys=labelSys)
    listAnswer = []
    listAnswer.append(answerSys)
    resultSys = ResultSys(id=id_result,imageSys=imageSys,listAnswer=listAnswer)
    # Trả về thông báo thành công
    return json.dumps(resultSys,default=lambda obj: obj.__dict__)