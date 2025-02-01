# BITCOIN-PRICE-PREDICT
# Bitcoin Price Prediction

## Overview
This project aims to predict Bitcoin prices using machine learning techniques. It utilizes historical price data and various technical indicators to build a predictive model that forecasts future Bitcoin prices.

## Features
- Data collection from cryptocurrency APIs
- Data preprocessing and feature engineering
- Model selection and training (e.g., Linear Regression, LSTM, Random Forest, etc.)
- Performance evaluation and visualization
- Deployment as a web app (optional)

## Technologies Used
- Python
- Pandas, NumPy for data manipulation
- Scikit-Learn, TensorFlow/Keras for machine learning models
- Matplotlib, Seaborn for visualization
- Flask/Streamlit for web deployment (if applicable)
- IBM Cloud Watson Machine Learning API for model deployment

## Installation
### Prerequisites
Ensure you have Python 3.x installed along with the required dependencies.

### Clone the Repository
```bash
git clone https://github.com/yourusername/bitcoin-price-prediction.git
cd bitcoin-price-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Data Preparation
1. Collect Bitcoin price data from sources like Yahoo Finance or Binance API.
2. Store data in CSV format inside the `data/` folder.
3. Run the preprocessing script:
```bash
python preprocess.py
```

### Training the Model
Run the model training script:
```bash
python train.py
```
This will generate a trained model saved in the `models/` directory.

### Deploying the Model on IBM Cloud
1. Set up your IBM Cloud account and get your API key.
2. Use the provided Python script to authenticate with IBM Cloud Watson ML API:
```python
import requests

API_KEY = "<your API key>"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
```
3. Define your input fields and values for model scoring:
```python
payload_scoring = {"input_data": [{"fields": [array_of_input_fields], "values": [array_of_values_to_be_scored]}]}
```
4. Send a request to IBM Watson ML API for prediction:
```python
response_scoring = requests.post('https://private.eu-gb.ml.cloud.ibm.com/ml/v4/deployments/<deployment_id>/predictions?version=2021-05-01', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
print(response_scoring.json())
```

### Making Predictions
To predict Bitcoin prices locally:
```bash
python predict.py --input sample_data.csv
```

## Results
- Model performance metrics (RMSE, R² Score)
- Price trend predictions plotted using Matplotlib/Seaborn

## Future Improvements
- Integration of real-time Bitcoin price streaming
- Hyperparameter tuning for better accuracy
- Deploying the model as an API
- Improved cloud integration and automation

## Contributing
Feel free to fork this repository and submit pull requests for enhancements or bug fixes.

## License
This project is licensed under the MIT License.

