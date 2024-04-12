import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://attendance-c56bd-default-rtdb.firebaseio.com/"
})

ref = db.reference('Member')

data = {
    "21521048": 
        {
            "Name": "Nguyen Tran Gia Ky",
            "ID": "21521048",
            "Total attendance": 7,
            "last attendance time": "2023-11-19 00:00:00"
        },
    "21523852":
        {
            "Name": "Elon Musk",
            "ID": "21523852",
            "Total attendance": 8,
            "last attendance time": "2023-11-20 01:01:01"
        }
}

for key, value in data.items():
    ref.child(key).set(value)
