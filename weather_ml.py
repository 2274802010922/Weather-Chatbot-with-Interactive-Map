import requests
import pandas as pd
import joblib

API_KEY = "703d35b679aa8f01b02eff474186a6b0"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


def get_weather_data(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "vi"
    }

    response = requests.get(FORECAST_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    rows = []
    for item in data["list"]:
        rows.append({
            "datetime": item["dt_txt"],
            "temp": item["main"]["temp"],
            "humidity": item["main"]["humidity"],  # 0-100
            "wind_speed": item["wind"]["speed"] * 3.6,  # m/s -> km/h
            "wind_bearing": item["wind"].get("deg", 0),
            "visibility": item.get("visibility", 10000) / 1000,  # m -> km
            "pressure": item["main"]["pressure"]
        })

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    return df


def predict_future_temperature(df):
    model = joblib.load("weather_model.pkl")

    feature_columns = [
        "humidity",
        "wind_speed",
        "wind_bearing",
        "visibility",
        "pressure",
        "hour",
        "day",
        "month",
        "year"
    ]

    prediction_df = df.copy()
    prediction_df["Predicted Temperature"] = model.predict(prediction_df[feature_columns])

    return prediction_df[["datetime", "temp", "Predicted Temperature"]]
