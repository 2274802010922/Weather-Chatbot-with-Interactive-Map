
import requests
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

API_KEY = "703d35b679aa8f01b02eff474186a6b0"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

CITIES = ["Hanoi", "Ho Chi Minh City", "Da Nang", "Can Tho", "Hai Phong"]


def get_forecast_data(city):
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
            "city": city,
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
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month

    # target: nhiệt độ hiện tại
    # feature: các biến thời tiết + thời gian
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
    print("Building dataset...")
    df = build_dataset(CITIES)

    # lưu data thô để sau này dùng cho Data Science
    df.to_csv("weather_dataset.csv", index=False)
    print("Saved dataset to weather_dataset.csv")

    X, y, df = add_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=42))
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
    ])

    print("\nModel comparison:")
    print(results_df.sort_values(by="RMSE"))

    results_df.to_csv("model_comparison.csv", index=False)
    print("Saved model comparison to model_comparison.csv")

    best_result = min(results, key=lambda x: x["rmse"])
    best_model = best_result["model"]

    joblib.dump(best_model, "weather_model.pkl")
    print(f"Best model: {best_result['name']}")
    print("Saved best model to weather_model.pkl")


if __name__ == "__main__":
    main()
