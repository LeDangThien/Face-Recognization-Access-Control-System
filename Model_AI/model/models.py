import cv2
import os
import urllib
import numpy as np
import pickle
import face_recognition
import math
import sys
import serial
import time
from datetime import datetime
import firebase_admin
from firebase_admin import storage
from firebase_admin import credentials
from firebase_admin import db
import string

cred = credentials.Certificate("model\serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://attendance-c56bd-default-rtdb.firebaseio.com/",
    'storageBucket': "facerecognitionrealtime-a3c69.appspot.com" 
})
bucket = storage.bucket()

# Replace the URL with the IP camera's stream URL
#url = 'http://192.168.137.210/cam-hi.jpg'
#cv2.namedWindow("live Cam Testing", cv2.WINDOW_AUTOSIZE)
 
# Create a VideoCapture object
#cap = cv2.VideoCapture(url)
# Check if the IP camera stream is opened successfully
# if not cap.isOpened():
#     print("Failed to open the IP camera stream")
#     exit()

#Load the encoding file 
print("Loading Encoded File...")
file = open('model\EncodeFile.p', 'rb')
encodeListKnownWithID = pickle.load(file)
file.close()
encodeListKnown, peopleID = encodeListKnownWithID
print(peopleID)
print("Encoded File Loaded")

# def lcd(i):
#     ok = b'Welcome'
#     eror = b'Error'
#     ser = serial.Serial('COM5', baudrate= 9600)
#     if i == 1:
#         ser.write(ok)
#     if i == 0:
#         ser.write(eror)
#     return ser

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance)/(range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + (1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2)) * 100
        return str(round(value, 2)) + '%'
    
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frames = True
    faceRegCheck = 0
    id = []
    peopleInfo = {"Full name": "", "Age": 0, "Phone number": "", "Last attendance": ""}
    def __init__(self):
        self.encode_faces()
        #encode faces

    def encode_faces(self):
        for image in os.listdir('Images'):
            face_image = face_recognition.load_image_file(f'Images/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        #print(self.known_face_names)    #in ra list các tên ảnh

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture:
            sys.exit('Video not found...')
        
        while True:
            ret, frame = video_capture.read()
            # img_resp= urllib.request.urlopen(url)
            # imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
            # frame = cv2.imdecode(imgnp,-1)
            # frame = cv2.resize(frame, (640, 480))
            if not ret:
                break
            else:
                if self.process_current_frames:
                    small_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    #find all faces in current frame
                    self.face_locations = face_recognition.face_locations(rgb_small_frame)
                    self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                    if len(self.face_locations) == 0:
                        self.faceRegCheck = 0
                        #lcd(0)
                    if len(self.face_locations) >= 2:
                        print('There are too many people in front of the camera')
                        self.faceRegCheck = 3
                        

                    self.face_names = []
                    for face_encoding in self.face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.45)
                        name = 'Unknown'
                        confidence = 'Unknown'
                        self.faceRegCheck = 2
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            confidence = face_confidence(face_distances[best_match_index])
                            self.id = peopleID[best_match_index]
                            self.peopleInfo = db.reference(f'Users/{self.id}').get()
                            #name = self.known_face_names[best_match_index]
                            name = str(self.peopleInfo["Full name"])
                            self.faceRegCheck = 1
                            #print(face_distances)
                        #self.face_names.append(f'{name} ({confidence})') #in tên file ảnh và khoảng tin cậy

                        #get the data
                        #peopleInfo = db.reference(f'Users/{self.id}').get() #Lấy thông tin từ các key 'id' thuộc lớp 'Users'
                        #print(peopleInfo)
                        #update data of "Total attendance"
                        #dateTimeObject = datetime.strptime(peopleInfo['last attendance time'], "%Y-%m-%d %H:%M:%S")
                        #Tính thời gian trôi qua kể từ khi nhận dạng được khuôn mặt
                        #secondElapsed =  (datetime.now() - dateTimeObject).total_seconds()
                        #print(secondElapsed)
                        #name = str(self.peopleInfo["Full name"])
                        self.face_names.append(f'{name} ({confidence})')

                self.process_current_frames = not self.process_current_frames

                #Display annotations
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4   
                    name = string.capwords(name)
                    cv2.rectangle(frame, (left + 1, top + 1), (right + 1, bottom + 1), (0, 0, 255), 1)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                    cv2.putText(frame, name, (left - 100, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                
                #print(self.faceRegCheck)                
                #ret,buffer = cv2.imencode('.jpg', frame)
                #frame = buffer.tobytes()
                cv2.imshow('Face Recognition', frame)
                #if self.faceRegCheck == 1:
                    #time.sleep(2)
                    # lcd(1) 
                    # time.sleep(0.5)
                    # lcd(0)
                    # # #print(self.faceRegCheck)
                    # time.sleep(0.5) 
            #yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
                if cv2.waitKey(1) == ord('q'):
                    break
            #video_capture.release()
            #cv2.destroyAllWindows()   
                        
if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()