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

#Load the encoding file 
print("Loading Encoded File...")
file = open('model\EncodeFile.p', 'rb')
encodeListKnownWithID = pickle.load(file)
file.close()
encodeListKnown, peopleID = encodeListKnownWithID
print(peopleID)
print("Encoded File Loaded")
