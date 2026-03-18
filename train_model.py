import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "weatherHistory.csv"


def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    selected_columns = [
        "Formatted Date",
        "Temperature (C)",
        "Apparent Temperature (C)",
        "Humidity",
        "Wind Speed (km/h)",
        "Wind Bearing (degrees)",
        "Visibility (km)",
        "Pressure (millibars)"
    ]

    df = df[selected_columns].copy()

    df.columns = [
        "datetime",
        "temp",
        "feels_like",
        "humidity",
        "wind_speed",
        "wind_bearing",
        "visibility",
        "pressure"
    ]

    df = df.dropna()

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Đưa humidity về cùng đơn vị với API OpenWeatherMap (0-100)
    df["humidity"] = df["humidity"] * 100

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    return df


def prepare_features(df):
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

    X = df[feature_columns]
    y = df["temp"]
    return X, y


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
    print("Loading dataset from Kaggle CSV...")
    df = load_and_clean_data(DATA_PATH)

    df.to_csv("weather_dataset_cleaned.csv", index=False)
    print("Saved cleaned dataset to weather_dataset_cleaned.csv")

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(
            n_estimators=20,
            max_depth=5,
            random_state=42
        )),
        ("Gradient Boosting", GradientBoostingRegressor(
            n_estimators=50,
            random_state=42
        ))
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
