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

classes_backup = { 0:'Speed limit (20km/h)',
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
    try:
        image = request.files["image"]
        id_result = int(request.form.get("id"))
        image_path = request.form.get("path")
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
        labelSys = None
        try:
            labelSys = list(filter(lambda item: item.id == (index_label+1),classes))[0]
        except:
            id_label = int(max_index)
            name_label = classes_backup.get(id_label)
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%d-%m-%Y")
            labelSys = LabelSys(id=id_label,name=name_label,dateEdit=str(formatted_time))
        # xử lý kết quả
        imageSys = ImageSys(id=id_result,path=image_path) 
        answerSys = AnswerSys(id=id_result,boxImage=boxImage,labelSys=labelSys)
        listAnswer = []
        listAnswer.append(answerSys)
        resultSys = ResultSys(id=id_result,imageSys=imageSys,listAnswer=listAnswer)
        # Trả về thông báo thành công
        return json.dumps(resultSys,default=lambda obj: obj.__dict__)
    except:
        return "An unknown error"