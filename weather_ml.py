import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


API_KEY = "YOUR_OPENWEATHER_API_KEY"


def get_weather_data(city):

    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    temps = []
    times = []

    for item in data["list"]:
        temps.append(item["main"]["temp"])
        times.append(item["dt"])

    df = pd.DataFrame({
        "time": times,
        "temperature": temps
    })

    return df


def predict_future_temperature(df):

    X = np.arange(len(df)).reshape(-1,1)
    y = df["temperature"]

    model = LinearRegression()
    model.fit(X,y)

    future = np.arange(len(df), len(df)+5).reshape(-1,1)

    predictions = model.predict(future)

    future_df = pd.DataFrame({
        "Step": range(1,6),
        "Predicted Temperature": predictions
    })

    return future_df
