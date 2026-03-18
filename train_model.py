import requests
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

API_KEY = "703d35b679aa8f01b02eff474186a6b0"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Dùng nhiều thành phố Việt Nam để model học đúng khí hậu hơn
CITIES = [
    "Hanoi",
    "Ho Chi Minh City",
    "Da Nang",
    "Can Tho",
    "Nha Trang",
    "Hue",
    "Vung Tau",
    "Quy Nhon",
    "Phan Thiet",
    "Buon Ma Thuot",
    "Pleiku",
    "Vinh",
    "Thai Nguyen",
    "Hai Duong",
    "Nam Dinh",
    "Long Xuyen",
    "Rach Gia",
    "Ca Mau",
    "Ha Long",
    "Bien Hoa"
]


def get_forecast_data(city):
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
            "city": city,
            "datetime": item["dt_txt"],
            "temp": item["main"]["temp"],
            "humidity": item["main"]["humidity"],                # %
            "pressure": item["main"]["pressure"],                # hPa
            "wind_speed": item["wind"]["speed"],                 # m/s
            "wind_bearing": item["wind"].get("deg", 0),          # độ
            "visibility": item.get("visibility", 10000),         # mét
            "clouds": item["clouds"]["all"],                     # %
            "lat": lat,
            "lon": lon
        })

    return pd.DataFrame(rows)


def build_dataset(cities):
    all_data = []

    for city in cities:
        try:
            df_city = get_forecast_data(city)
            all_data.append(df_city)
            print(f"Loaded forecast data for {city}")
        except Exception as e:
            print(f"Error loading data for {city}: {e}")

    if not all_data:
        raise ValueError("Không lấy được dữ liệu từ city nào.")

    df = pd.concat(all_data, ignore_index=True)
    return df


def add_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month

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

    X = df[feature_columns]
    y = df["temp"]

    return X, y, df


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    return {
        "name": name,
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def main():
    print("Building API-based weather dataset...")
    df = build_dataset(CITIES)

    df.to_csv("weather_api_training_dataset.csv", index=False)
    print("Saved dataset to weather_api_training_dataset.csv")

    X, y, df = add_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = [
        ("Linear Regression", LinearRegression()),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=80,
                max_depth=8,
                random_state=42
            )
        ),
        (
            "Gradient Boosting",
            GradientBoostingRegressor(
                n_estimators=120,
                random_state=42
            )
        )
    ]

    results = []
    for name, model in models:
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(result)

    results_df = pd.DataFrame([
        {
            "Model": r["name"],
            "MAE": r["mae"],
            "RMSE": r["rmse"],
            "R2": r["r2"]
        }
        for r in results
    ]).sort_values(by="RMSE")

    print("\nModel comparison:")
    print(results_df)

    results_df.to_csv("model_comparison.csv", index=False)
    print("Saved model comparison to model_comparison.csv")

    best_result = min(results, key=lambda x: x["rmse"])
    best_model = best_result["model"]

    joblib.dump(best_model, "weather_model.pkl")
    print(f"Best model: {best_result['name']}")
    print("Saved best model to weather_model.pkl")


if __name__ == "__main__":
    main()
