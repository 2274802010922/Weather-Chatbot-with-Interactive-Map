import requests
import pandas as pd
import joblib

API_KEY = "703d35b679aa8f01b02eff474186a6b0"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


def get_weather_data(city):
    url = f"{FORECAST_URL}?q={city}&appid={API_KEY}&units=metric&lang=vi"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    rows = []
    for item in data["list"]:
        rows.append({
            "datetime": item["dt_txt"],
            "temp": item["main"]["temp"],
            "feels_like": item["main"]["feels_like"],
            "temp_min": item["main"]["temp_min"],
            "temp_max": item["main"]["temp_max"],
            "pressure": item["main"]["pressure"],
            "humidity": item["main"]["humidity"],
            "wind_speed": item["wind"]["speed"],
            "clouds": item["clouds"]["all"]
        })

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    return df


def predict_future_temperature(df):
    model = joblib.load("weather_model.pkl")

    feature_columns = [
        "feels_like",
        "temp_min",
        "temp_max",
        "pressure",
        "humidity",
        "wind_speed",
        "clouds",
        "hour",
        "day",
        "month"
    ]

    prediction_df = df.copy()
    prediction_df["Predicted Temperature"] = model.predict(prediction_df[feature_columns])

    return prediction_df[["datetime", "temp", "Predicted Temperature"]]
