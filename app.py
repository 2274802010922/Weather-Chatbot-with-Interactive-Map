import re
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

from weather_ml import get_weather_data, predict_future_temperature

API_KEY = "703d35b679aa8f01b02eff474186a6b0"
CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


st.set_page_config(page_title="Weather Chatbot with ML", layout="wide")
st.title("🌦️ Weather Chatbot with Interactive Map + ML Dashboard")


def extract_city(user_input: str) -> str:
    if not user_input:
        return ""

    text = user_input.strip()
    patterns = [
        r"thời tiết ở (.+)",
        r"thời tiết tại (.+)",
        r"thoi tiet o (.+)",
        r"thoi tiet tai (.+)",
        r"weather in (.+)",
        r"forecast in (.+)",
    ]

    lowered = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return match.group(1).strip(" ?,.!").title()

    return text.strip(" ?,.!").title()


def get_current_weather(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "vi"
    }

    response = requests.get(CURRENT_WEATHER_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def build_map_df(current_data):
    coord = current_data["coord"]
    return pd.DataFrame({
        "lat": [coord["lat"]],
        "lon": [coord["lon"]]
    })


tab1, tab2, tab3 = st.tabs([
    "💬 Chatbot & Current Weather",
    "🤖 ML Prediction",
    "📊 Data Science Dashboard"
])

with tab1:
    st.subheader("Nhập câu hỏi hoặc tên thành phố")
    user_input = st.text_input(
        "Ví dụ: thời tiết ở Hà Nội / weather in Da Nang",
        value="Hanoi"
    )

    if user_input:
        city = extract_city(user_input)

        try:
            current_data = get_current_weather(city)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Thành phố", current_data["name"])
                st.metric("Nhiệt độ", f'{current_data["main"]["temp"]:.1f} °C')

            with col2:
                st.metric("Độ ẩm", f'{current_data["main"]["humidity"]}%')
                st.metric("Áp suất", f'{current_data["main"]["pressure"]} hPa')

            with col3:
                st.metric("Tốc độ gió", f'{current_data["wind"]["speed"]:.1f} m/s')
                st.metric("Mô tả", current_data["weather"][0]["description"])

            st.subheader("🗺️ Interactive Map")
            st.map(build_map_df(current_data), zoom=8)

        except Exception as e:
            st.error(f"Không lấy được dữ liệu thời tiết hiện tại: {e}")

with tab2:
    st.subheader("Dự đoán nhiệt độ bằng mô hình Machine Learning")

    city_for_prediction = st.text_input("Tên thành phố để dự đoán", value="Hanoi", key="prediction_city")

    if st.button("Chạy dự đoán"):
        try:
            forecast_df = get_weather_data(city_for_prediction)
            prediction_df = predict_future_temperature(forecast_df)

            st.write("### Bảng dự đoán")
            st.dataframe(prediction_df, use_container_width=True)

            fig = px.line(
                prediction_df,
                x="datetime",
                y=["temp", "Predicted Temperature"],
                title="So sánh nhiệt độ thực tế và nhiệt độ dự đoán"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write("### Đánh giá mô hình")
            model_df = pd.read_csv("model_comparison.csv")
            st.dataframe(model_df, use_container_width=True)

        except FileNotFoundError:
            st.error("Không tìm thấy weather_model.pkl hoặc model_comparison.csv. Hãy chắc rằng bạn đã upload đủ file lên GitHub.")
        except Exception as e:
            st.error(f"Lỗi khi chạy dự đoán: {e}")

with tab3:
    st.subheader("Phân tích dữ liệu từ Kaggle")

    try:
        df = pd.read_csv("weather_dataset_cleaned.csv")

        st.write("### Xem nhanh dữ liệu")
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            fig_temp = px.histogram(
                df,
                x="temp",
                nbins=40,
                title="Phân bố nhiệt độ"
            )
            st.plotly_chart(fig_temp, use_container_width=True)

        with col2:
            fig_humidity = px.histogram(
                df,
                x="humidity",
                nbins=40,
                title="Phân bố độ ẩm"
            )
            st.plotly_chart(fig_humidity, use_container_width=True)

        sample_df = df.head(300).copy()
        fig_line = px.line(
            sample_df,
            x="datetime",
            y="temp",
            title="Biến động nhiệt độ theo thời gian (300 dòng đầu)"
        )
        st.plotly_chart(fig_line, use_container_width=True)

        corr_cols = [
            "temp",
            "humidity",
            "wind_speed",
            "wind_bearing",
            "visibility",
            "pressure"
        ]
        corr_df = df[corr_cols].corr().round(2)

        fig_corr = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            title="Ma trận tương quan"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    except FileNotFoundError:
        st.error("Không tìm thấy weather_dataset_cleaned.csv. Hãy upload file này lên GitHub.")
    except Exception as e:
        st.error(f"Lỗi khi hiển thị dashboard dữ liệu: {e}")
