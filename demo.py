import pandas as pd
import requests

# Load dataset
df = pd.read_csv('./data/credit_pred.csv')

# Define the API endpoint URL
api_url = "http://127.0.0.1:8000/predict"

# Prepare an empty list to store predictions
predictions = []
payload = {}
# Iterate through each row and send a request to the FastAPI prediction API
for _, row in df.iterrows():
    # Prepare the data payload (convert the row to a dictionary)
    payload = [row["X"+ str(x+1)].item() for x in range(23)]

    # Send POST request to FastAPI /predict endpoint
    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        # Extract the prediction from the response
        prediction = response.json()['prediction']
        predictions.append(prediction)
    else:
        print(f"Failed to get prediction for row {_}. Status code: {response.status_code}")
        predictions.append(None)

# Add predictions to the DataFrame
df['Y'] = predictions

# Save the dataset with predictions
df.to_csv('./data/credit_pred_with_predictions.csv', index=False)
print("Predictions added to 'credit_pred_with_predictions.csv'")
