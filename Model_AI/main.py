from flask import Blueprint, render_template, request, redirect, url_for, Response, Flask
import pyrebase
import cv2
import face_recognition
from datetime import datetime
import os
import urllib
import sys
import pickle
import math
import time
import numpy as np
import firebase_admin
from firebase_admin import storage
from firebase_admin import credentials
from firebase_admin import db   
from flask_socketio import SocketIO, emit
import threading
import serial
#from model import EncodeGenerator

cred = credentials.Certificate("model\serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://attendance-c56bd-default-rtdb.firebaseio.com/",
    'storageBucket': "attendance-c56bd.appspot.com" 
})
bucket = storage.bucket()

#Import the people images to the list
# folderPath = 'Images'
# pathList = os.listdir(folderPath)
# imgList = []
# peopleID = []

# for path in pathList:
#     #Hàm append: thêm 1 đối tượng vào danh sách
#     imgList.append(cv2.imread(os.path.join(folderPath, path)))
#     peopleID.append(os.path.splitext(path)[0])

#     fileName = f'{folderPath}/{path}'
#     #tạo bucket để chứa ảnh
#     bucket = storage.bucket()
#     blob = bucket.blob(fileName)
#     blob.upload_from_filename(fileName)

#     print(path)
#     print(os.path.splitext(path)[0])
# print(peopleID) 
    
#Encoding progress of CNN encodes the image to 128-dimensions (Facenet Algorithm)
# def findEncodings(imagesList):
#     encodeList = []
#     for img in imagesList:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]    
#         encodeList.append(encode)
#     return encodeList
# print("Encoding starting...")
# encodeListKnown = findEncodings(imgList)
# encodeListKnownWithID = [encodeListKnown, peopleID]
# #print(encodeListKnown)
# print("Encoding completed")

#Import to pickle file
# file = open("model\EncodeFile.p", 'wb')
# pickle.dump(encodeListKnownWithID, file)
# file.close()
# print("File saved")

app=Flask(__name__)
app.config['SECET_KEY']='CNIASUINC'

socketio=SocketIO(app)


config={
    "apiKey": "AIzaSyCXBQNcm5P0HyMKt7V1JYm7Ruxji9LPUms",
    "authDomain": "attendance-c56bd.firebaseapp.com",
    "databaseURL": "https://attendance-c56bd-default-rtdb.firebaseio.com",
    "storageBucket": "attendance-c56bd.appspot.com/Images",
}   

firebase = pyrebase.initialize_app(config)
authFb = firebase.auth()
database = firebase.database()


# #Load the encoding file 
# print("Loading Encoded File...")
# file = open('model\EncodeFile.p', 'rb')
# encodeListKnownWithID = pickle.load(file)
# file.close()
# encodeListKnown, peopleID = encodeListKnownWithID
# print(peopleID)
# print("Encoded File Loaded")

def face_confidence(face_distance, face_match_threshold=0.4):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance)/(range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + (1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2)) * 100
        return str(round(value, 2)) + '%'

def lcd(i):
    ok = b'Welcome'
    eror = b'Error'
    ser = serial.Serial('COM5', baudrate= 9600)
    if i == 1:
        ser.write(ok)
    if i == 0:
        ser.write(eror)
    return ser

url = 'http://192.168.137.238/cam-hi.jpg'
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frames = True
    faceRegCheck = 0
    id = []
    peopleID = []
    peopleInfo = {"Full name": "", "Age": 0, "Phone number": "", "Last attendance": ""}
    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        # for image in os.listdir('Images'):
        #     face_image = face_recognition.load_image_file(f'Images/{image}')
        #     face_encoding = face_recognition.face_encodings(face_image)[0]
        #     self.known_face_encodings.append(face_encoding)
        #     self.known_face_names.append(image)
        #Load the encoding file 
        print("Loading Encoded File...")
        file = open('model\EncodeFile.p', 'rb')
        encodeListKnownWithID = pickle.load(file)
        file.close()
        encodeListKnown, self.peopleID = encodeListKnownWithID
        print(self.peopleID)
        print("Encoded File Loaded")
        #print(encodeListKnownWithID)
        self.known_face_encodings, self.known_face_names = encodeListKnownWithID
        #print(self.known_face_names)    #in ra list các tên ảnh
    
    def run_recognition(self):
        #video_capture = cv2.VideoCapture(url)
        # video_capture = cv2.VideoCapture(0)

        # if not video_capture:
        #     sys.exit('Video not found...')
        
        while True:
            # ret, frame = video_capture.read()
            # Read a frame from the video stream
            img_resp= urllib.request.urlopen(url)
            imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
            frame = cv2.imdecode(imgnp,-1)
            frame = cv2.resize(frame, (640, 480))
            if self.process_current_frames:
                small_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                #find all faces in current frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                if len(self.face_locations) == 0:
                    self.faceRegCheck = 0
                    lcd(0)
                    #self.peopleInfo['Age'] = -1
                if len(self.face_locations) >= 2:
                    print('There are too many people in front of the camera')
                    self.faceRegCheck = 3
                    self.peopleInfo['Age'] = -2
                    socketio.emit('faceRegCheck_update', {'peopleID': self.peopleInfo})
                    time.sleep(1)
                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.4)
                    name = 'Unknown'
                    confidence = 'Unknown'
                    self.faceRegCheck = 2
                    #self.peopleInfo['Age'] = -1
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                        self.id = self.peopleID[best_match_index]
                        self.faceRegCheck = 1
                        self.peopleInfo = db.reference(f'Users/{self.id}').get() #Lấy thông tin từ các key 'id' thuộc lớp 'Users'
                        #print(self.peopleInfo)
                        name = str(self.peopleInfo["Full name"])
                    #self.face_names.append(f'{name} ({confidence})') #in tên file ảnh và khoảng tin cậy
                    #update data of "Total attendance"
                    #dateTimeObject = datetime.strptime(peopleInfo['last attendance time'], "%Y-%m-%d %H:%M:%S")
                    #Tính thời gian trôi qua kể từ khi nhận dạng được khuôn mặt
                    #secondElapsed =  (datetime.now() - dateTimeObject).total_seconds()
                    #print(secondElapsed)
                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frames = not self.process_current_frames

            #Display annotations
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left + 1, top + 1), (right + 1, bottom + 1), (0, 0, 255), 1)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            #print(self.faceRegCheck)                
            ret,buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            #cv2.imshow('Face Recognition', frame)
            if self.faceRegCheck == 1:
                socketio.emit('faceRegCheck_update', {'peopleID': self.peopleInfo})
                time.sleep(1)
                lcd(1) 
                # lcd(0)
                #time.sleep(2)                  
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

personToDb={"Full name": "", "Age": 0, "Phone number": ""}  # thiếu hình
#nếu ko có ai trước camera thì faceRegCheck=0
#nếu nhận diện đúng khuôn mặt thì faceRegCheck=1
#nếu nhận diện ko đúng thì faceRegCheck=2

@app.route('/page0')
def page0():
    return render_template("page0.html")
#cap = cv2.VideoCapture(0)
def generate_frames():
    while True:
       
            # small_frame = cv2.resize(img, (0, 0), fx = 0.25, fy = 0.25)
            # rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            # face_loc = face_recognition.face_locations(rgb_small_frame, model="hog")
            # top, right, bot, top = face_loc
            # top *= 4
            # right *= 4
            # bottom *= 4
            # left *= 4   
            # cv2.rectangle(img, (left + 1, top + 1), (right + 1, bottom + 1), (0, 0, 255), 1)
        img_resp= urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        frame = cv2.imdecode(imgnp,-1)
        frame = cv2.resize(frame, (640, 480))
        ret, buffer = cv2.imencode('.jpg', frame)
        img = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

def capture_image(id):
    #_, frame = cap.read()
    img_resp= urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    frame = cv2.imdecode(imgnp,-1)
    frame = cv2.resize(frame, (640, 480))
    ret, buffer = cv2.imencode('.jpg', frame)
    img = buffer.tobytes()
    folder_path = 'Images'
    image_path = os.path.join(folder_path, f'{id}.png')
    cv2.imwrite(image_path, frame)
    return image_path

@app.route('/webcam')
def webcam():     
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method=='POST':
        name=request.form.get('name')
        age=request.form.get('age')
        phone=request.form.get('phone')
        global personToDb
        personToDb["Full name"]=name
        personToDb["Age"]=age
        personToDb["Phone number"]=phone
        #personToDb["Last attendance"] = set(datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        id = phone[(len(phone) - 4) : len(phone)]
        capture_image(id)
        
        #Encoding progress of CNN encodes the image to 128-dimensions (Facenet Algorithm)
        def findEncodings(imagesList):
            encodeList = []
            for img in imagesList:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]    
                encodeList.append(encode)
            return encodeList
        
        folderPath = 'Images'
        pathList = os.listdir(folderPath)
        imgList = []
        peopleID = []

        for path in pathList:
            #Hàm append: thêm 1 đối tượng vào danh sách
            imgList.append(cv2.imread(os.path.join(folderPath, path)))
            peopleID.append(os.path.splitext(path)[0])

            fileName = f'{folderPath}/{path}'
            #tạo bucket để chứa ảnh
            bucket = storage.bucket()
            blob = bucket.blob(fileName)
            blob.upload_from_filename(fileName)

        print("Encoding starting...")
        encodeListKnown = findEncodings(imgList)
        encodeListKnownWithID = [encodeListKnown, peopleID]
        #print(encodeListKnown)
        print("Encoding completed")

        #Import to pickle file
        file = open("model\EncodeFile.p", 'wb')
        pickle.dump(encodeListKnownWithID, file)
        file.close()
        print("File saved")

        database.child("Users").child(id).set(personToDb)
    return render_template("page0.html")

fr = FaceRecognition()
@app.route('/video')
def video(): 
    return Response(fr.run_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    fr.encode_faces()    
    return render_template("home.html")
if __name__=='__main__':
    #app.run(debug=True)
    socketio.run(app,debug=True)