import requests

url = 'http://127.0.0.1:8000/predict'
data = [x for x in range(23)]

print(data)

response = requests.post(url, json=data)
print(f"Prediction: {response.json()}")
