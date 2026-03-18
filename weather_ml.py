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

    response = requests.get(FORECAST_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    city_coord = data.get("city", {}).get("coord", {})
    lat = city_coord.get("lat", 0.0)
    lon = city_coord.get("lon", 0.0)

    rows = []
    for item in data["list"]:
        rows.append({
            "datetime": item["dt_txt"],
            "temp": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "pressure": item["main"]["pressure"],
            "wind_speed": item["wind"]["speed"],
            "wind_bearing": item["wind"].get("deg", 0),
            "visibility": item.get("visibility", 10000),
            "clouds": item["clouds"]["all"],
            "lat": lat,
            "lon": lon
        })

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month

    return df


def predict_future_temperature(df):
    model = joblib.load("weather_model.pkl")

    feature_columns = [
        "humidity",
        "pressure",
        "wind_speed",
        "wind_bearing",
        "visibility",
        "clouds",
        "lat",
        "lon",
        "hour",
        "day",
        "month"
    ]

    prediction_df = df.copy()
    prediction_df["Estimated Temperature (ML)"] = model.predict(
        prediction_df[feature_columns]
    )

    return prediction_df[["datetime", "temp", "Estimated Temperature (ML)"]]
