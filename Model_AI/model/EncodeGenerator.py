import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import storage
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("model\serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://attendance-c56bd-default-rtdb.firebaseio.com/",
    'storageBucket': "facerecognitionrealtime-a3c69.appspot.com" 
})

#Import the people images to the list
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

    #print(path)
    #print(os.path.splitext(path)[0])
#print(peopleID) 

#Encoding progress of CNN encodes the image to 128-dimensions (Facenet Algorithm)
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]    
        encodeList.append(encode)
    return encodeList
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
