import re
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import folium

from folium.plugins import MiniMap, Fullscreen
from streamlit_folium import st_folium
from weather_ml import get_weather_data, predict_future_temperature

API_KEY = "703d35b679aa8f01b02eff474186a6b0"
CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

st.set_page_config(page_title="Weather Chatbot + ML Dashboard", layout="wide")

st.title("🌦️ Weather Chatbot with Interactive Map + ML Dashboard")
st.caption(
    "Ứng dụng kết hợp thời tiết hiện tại, bản đồ lớp thời tiết trực quan, "
    "dự đoán nhiệt độ bằng Machine Learning và dashboard phân tích dữ liệu."
)


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

    response = requests.get(CURRENT_WEATHER_URL, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def create_weather_layer_map(lat, lon, city_name, current_temp):
    m = folium.Map(
        location=[lat, lon],
        zoom_start=7,
        tiles="CartoDB dark_matter"
    )

    folium.Marker(
        [lat, lon],
        tooltip=f"{city_name}: {current_temp:.1f} °C",
        popup=(
            f"<b>{city_name}</b><br>"
            f"Nhiệt độ hiện tại: {current_temp:.1f} °C<br>"
            f"Đây là vị trí trung tâm thành phố bạn đang xem."
        ),
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Lớp nhiệt độ
    folium.TileLayer(
        tiles=f"https://tile.openweathermap.org/map/temp_new/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
        attr="OpenWeatherMap",
        name="🌡️ Bản đồ nhiệt độ",
        overlay=True,
        control=True,
        opacity=0.75
    ).add_to(m)

    # Lớp mưa
    folium.TileLayer(
        tiles=f"https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
        attr="OpenWeatherMap",
        name="🌧️ Khu vực đang có mưa",
        overlay=True,
        control=True,
        opacity=0.75
    ).add_to(m)

    # Lớp gió
    folium.TileLayer(
        tiles=f"https://tile.openweathermap.org/map/wind_new/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
        attr="OpenWeatherMap",
        name="💨 Hướng và cường độ gió",
        overlay=True,
        control=True,
        opacity=0.75
    ).add_to(m)

    # Lớp mây
    folium.TileLayer(
        tiles=f"https://tile.openweathermap.org/map/clouds_new/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
        attr="OpenWeatherMap",
        name="☁️ Mật độ mây",
        overlay=True,
        control=True,
        opacity=0.75
    ).add_to(m)

    # Lớp áp suất
    folium.TileLayer(
        tiles=f"https://tile.openweathermap.org/map/pressure_new/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
        attr="OpenWeatherMap",
        name="🧭 Áp suất khí quyển",
        overlay=True,
        control=True,
        opacity=0.75
    ).add_to(m)

    MiniMap(toggle_display=True).add_to(m)
    Fullscreen().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def render_model_summary():
    model_df = pd.read_csv("model_comparison.csv")
    best_model = model_df.sort_values(by="RMSE").iloc[0]

    st.success(
        f"Mô hình tốt nhất hiện tại: {best_model['Model']} | "
        f"RMSE: {best_model['RMSE']:.3f} | "
        f"R²: {best_model['R2']:.3f}"
    )
    return model_df


tab1, tab2, tab3 = st.tabs([
    "💬 Chatbot & Thời tiết hiện tại",
    "🤖 Dự đoán ML",
    "📊 Dashboard dữ liệu"
])

with tab1:
    st.subheader("Nhập câu hỏi hoặc tên thành phố")
    st.caption(
        "Bạn có thể nhập kiểu: 'weather in Da Nang', "
        "hoặc chỉ nhập tên thành phố."
    )

    user_input = st.text_input(
        "Ví dụ:",
        value="ho chi minh"
    )

    if user_input:
        city = extract_city(user_input)

        try:
            current_data = get_current_weather(city)
            city_name = current_data["name"]
            temp = current_data["main"]["temp"]
            humidity = current_data["main"]["humidity"]
            pressure = current_data["main"]["pressure"]
            wind_speed = current_data["wind"]["speed"]
            description = current_data["weather"][0]["description"]
            lat = current_data["coord"]["lat"]
            lon = current_data["coord"]["lon"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📍 Thành phố", city_name)
                st.metric("🌡️ Nhiệt độ", f"{temp:.1f} °C")

            with col2:
                st.metric("💧 Độ ẩm", f"{humidity}%")
                st.metric("🧭 Áp suất", f"{pressure} hPa")

            with col3:
                st.metric("💨 Tốc độ gió", f"{wind_speed:.1f} m/s")
                st.metric("📝 Mô tả", description)

            st.subheader("🗺️ Bản đồ thời tiết nhiều lớp")
            st.info(
                "Chú thích: Bạn có thể bật/tắt các lớp ở góc phải bản đồ để xem "
                "nhiệt độ theo màu sắc khu vực, nơi đang có mưa, hướng gió, mây và áp suất."
            )

            weather_map = create_weather_layer_map(lat, lon, city_name, temp)
            st_folium(weather_map, width=None, height=550)

        except Exception as e:
            st.error(f"Không lấy được dữ liệu thời tiết hiện tại: {e}")

with tab2:
    st.subheader("Dự đoán nhiệt độ bằng mô hình Machine Learning")
    st.caption(
        "Phần này dùng mô hình đã train từ dataset Kaggle để ước lượng nhiệt độ "
        "dựa trên dữ liệu dự báo từ API."
    )

    city_for_prediction = st.text_input(
        "Tên thành phố để dự đoán",
        value="Hanoi",
        key="prediction_city"
    )

    if st.button("Chạy dự đoán"):
        try:
            forecast_df = get_weather_data(city_for_prediction)
            prediction_df = predict_future_temperature(forecast_df)

            st.info(
                "Chú thích: 'temp' là nhiệt độ từ dữ liệu dự báo API, "
                "còn 'Estimated Temperature (ML)' là nhiệt độ ước lượng từ mô hình học máy "
                "được train bằng dữ liệu OpenWeather API của nhiều thành phố Việt Nam."
            )

            st.write("### Bảng dự đoán")
            st.dataframe(prediction_df, use_container_width=True)

            fig = px.line(
                prediction_df,
                x="datetime",
                y=["temp", "Estimated Temperature (ML)"],
                title="So sánh nhiệt độ từ API và nhiệt độ ước lượng từ mô hình"
            )
            fig.update_layout(
                xaxis_title="Thời gian",
                yaxis_title="Nhiệt độ (°C)",
                legend_title="Biến"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write("### Đánh giá mô hình")
            model_df = render_model_summary()
            st.dataframe(model_df, use_container_width=True)

        except FileNotFoundError:
            st.error(
                "Không tìm thấy weather_model.pkl hoặc model_comparison.csv. "
                "Hãy chắc rằng bạn đã upload đủ file lên GitHub."
            )
        except Exception as e:
            st.error(f"Lỗi khi chạy dự đoán: {e}")

with tab3:
    st.subheader("Phân tích dữ liệu từ Kaggle")
    st.caption(
        "Tab này giúp người dùng hiểu dữ liệu lịch sử thời tiết dùng để train mô hình."
    )

    try:
        df = pd.read_csv("weather_dataset_cleaned.csv")

        st.write("### Xem nhanh dữ liệu")
        st.dataframe(df.head(20), use_container_width=True)

        st.info(
            "Chú thích: Dữ liệu đã được làm sạch và chuẩn hóa để huấn luyện mô hình. "
            "Độ ẩm ở đây đã được đổi về thang 0-100 để khớp với API."
        )

        col1, col2 = st.columns(2)

        with col1:
            fig_temp = px.histogram(
                df,
                x="temp",
                nbins=40,
                title="Phân bố nhiệt độ"
            )
            fig_temp.update_layout(
                xaxis_title="Nhiệt độ (°C)",
                yaxis_title="Số lượng mẫu"
            )
            st.plotly_chart(fig_temp, use_container_width=True)

        with col2:
            fig_humidity = px.histogram(
                df,
                x="humidity",
                nbins=40,
                title="Phân bố độ ẩm"
            )
            fig_humidity.update_layout(
                xaxis_title="Độ ẩm (%)",
                yaxis_title="Số lượng mẫu"
            )
            st.plotly_chart(fig_humidity, use_container_width=True)

        sample_df = df.head(300).copy()
        fig_line = px.line(
            sample_df,
            x="datetime",
            y="temp",
            title="Biến động nhiệt độ theo thời gian (300 dòng đầu)"
        )
        fig_line.update_layout(
            xaxis_title="Thời gian",
            yaxis_title="Nhiệt độ (°C)"
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
            title="Ma trận tương quan giữa các biến thời tiết"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    except FileNotFoundError:
        st.error(
            "Không tìm thấy weather_dataset_cleaned.csv. "
            "Hãy upload file này lên GitHub."
        )
    except Exception as e:
        st.error(f"Lỗi khi hiển thị dashboard dữ liệu: {e}")
