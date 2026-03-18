# Weather Chatbot with Interactive Map + Machine Learning

This project is a Streamlit web app that combines:

- Current weather chatbot
- Interactive map
- Machine learning temperature prediction
- Data Science dashboard using a Kaggle weather dataset

## Features

### 1. Current Weather Chatbot
- Ask about the weather by typing a city name
- Supports simple natural language input
- Shows:
  - city name
  - temperature
  - humidity
  - pressure
  - wind speed
  - weather description

### 2. Interactive Map
- Displays the selected city's location on the map

### 3. Machine Learning Prediction
- Uses a trained model to predict temperature from forecast data
- Compares:
  - actual temperature
  - predicted temperature

### 4. Data Science Dashboard
- Temperature distribution
- Humidity distribution
- Temperature trend chart
- Correlation heatmap
- Model evaluation table

## Machine Learning Models
The project compares:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Evaluation metrics:
- MAE
- RMSE
- R²

The best model is automatically selected and saved as:

- `weather_model.pkl`

## Files

- `app.py` → Streamlit app
- `weather_ml.py` → weather forecasting + ML prediction logic
- `train_model.py` → training pipeline using Kaggle dataset
- `weatherHistory.csv` → original Kaggle dataset
- `weather_dataset_cleaned.csv` → cleaned dataset
- `model_comparison.csv` → model evaluation results
- `weather_model.pkl` → trained best model

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
