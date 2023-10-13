import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'nitrogen':60, 'phorsporus':50, 'potassium':40,'temperature':20, 'humidity':80,'pH':7,'rainfall':200})

print(r.json())