Weather Chatbot with Interactive Map and Machine Learning Dashboard
Introduction

English:
This project is an interactive weather analytics application developed using Streamlit. It integrates real-time weather data retrieval, geospatial visualization, and machine learning techniques to estimate temperature from forecast data.

Tiếng Việt:
Dự án này là một ứng dụng phân tích thời tiết tương tác được xây dựng bằng Streamlit. Ứng dụng kết hợp việc lấy dữ liệu thời tiết thời gian thực, hiển thị bản đồ và sử dụng học máy để ước lượng nhiệt độ từ dữ liệu dự báo.

Key Features
Weather Chatbot

English:
The application supports natural language input such as "weather in Hanoi" or "Da Nang weather". It retrieves real-time weather data from the OpenWeatherMap API and displays key metrics including temperature, humidity, pressure, wind speed, and weather conditions.

Tiếng Việt:
Ứng dụng hỗ trợ nhập dữ liệu bằng ngôn ngữ tự nhiên như "weather in Hanoi" hoặc "thời tiết ở Đà Nẵng". Dữ liệu thời tiết được lấy từ OpenWeatherMap API và hiển thị các thông tin quan trọng như nhiệt độ, độ ẩm, áp suất, tốc độ gió và mô tả thời tiết.

Interactive Weather Map

English:
The system provides an interactive map built with Folium, allowing users to visualize multiple weather layers such as temperature, precipitation, wind, cloud coverage, and atmospheric pressure. Additional features include layer control, minimap, and fullscreen mode.

Tiếng Việt:
Hệ thống cung cấp bản đồ tương tác được xây dựng bằng Folium, cho phép người dùng xem nhiều lớp dữ liệu thời tiết như nhiệt độ, mưa, gió, mây và áp suất. Bản đồ hỗ trợ điều khiển lớp, minimap và chế độ toàn màn hình.

Machine Learning Prediction

English:
The application includes a machine learning module that estimates temperature based on forecast data. The feature set includes humidity, pressure, wind speed, wind direction, visibility, cloud coverage, geographical coordinates, and time-based variables such as hour, day, and month.

Multiple regression models were trained and evaluated, including Linear Regression, Random Forest, and Gradient Boosting. The best-performing model was selected based on RMSE and R squared metrics.

Tiếng Việt:
Ứng dụng tích hợp mô-đun học máy để ước lượng nhiệt độ dựa trên dữ liệu dự báo. Các đặc trưng sử dụng bao gồm độ ẩm, áp suất, tốc độ gió, hướng gió, tầm nhìn, mây, tọa độ địa lý và các yếu tố thời gian như giờ, ngày và tháng.

Nhiều mô hình hồi quy đã được huấn luyện và so sánh, bao gồm Linear Regression, Random Forest và Gradient Boosting. Mô hình tốt nhất được lựa chọn dựa trên RMSE và hệ số R bình phương.

Model Performance

English:
The Gradient Boosting model achieved the best performance among all evaluated models.

RMSE approximately 0.636
R squared approximately 0.969

Tiếng Việt:
Mô hình Gradient Boosting đạt hiệu suất tốt nhất trong các mô hình được thử nghiệm.

RMSE khoảng 0.636
R bình phương khoảng 0.969
Data Analytics Dashboard

English:
The application includes a dashboard for exploratory data analysis. It provides a dataset preview, distribution plots for temperature and humidity, time series visualization, and a correlation matrix.

Tiếng Việt:
Ứng dụng cung cấp dashboard để phân tích dữ liệu. Bao gồm xem dữ liệu mẫu, biểu đồ phân phối nhiệt độ và độ ẩm, biểu đồ xu hướng theo thời gian và ma trận tương quan.

Project Structure
Weather-Chatbot-with-Interactive-Map/
├── app.py
├── weather_ml.py
├── train_model.py
├── weather_model.pkl
├── model_comparison.csv
├── weather_dataset_cleaned.csv
├── weather_api_training_dataset.csv
├── weatherHistory.csv
├── requirements.txt
└── index.html
Technologies

English:
Python, Streamlit, Pandas, Plotly, Folium, Scikit-learn, Joblib, OpenWeatherMap API

Tiếng Việt:
Python, Streamlit, Pandas, Plotly, Folium, Scikit-learn, Joblib, OpenWeatherMap API

Installation

Step 1: Clone the repository
git clone https://github.com/2274802010922/Weather-Chatbot-with-Interactive-Map.git

cd Weather-Chatbot-with-Interactive-Map

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Run the application
streamlit run app.py

Usage

English:
Users can enter a city name or a natural language query to retrieve weather information. The application also allows users to explore forecast data, compare predicted and actual temperatures, and analyze historical datasets through the dashboard.

Tiếng Việt:
Người dùng có thể nhập tên thành phố hoặc câu hỏi để tra cứu thời tiết. Ứng dụng cũng cho phép xem dữ liệu dự báo, so sánh nhiệt độ dự đoán và thực tế, đồng thời phân tích dữ liệu thông qua dashboard.

Notes

English:
The API key is currently stored in the source code for demonstration purposes. It is recommended to move the API key to environment variables or a secure configuration file in production.

Tiếng Việt:
API key hiện đang được lưu trong mã nguồn để phục vụ mục đích demo. Trong môi trường thực tế, nên chuyển API key sang biến môi trường hoặc cấu hình bảo mật.

Future Improvements

English:
Future improvements may include integrating a more advanced conversational AI model, expanding dataset coverage, improving model performance, and deploying the application to a cloud platform.

Tiếng Việt:
Các hướng phát triển trong tương lai bao gồm tích hợp chatbot thông minh hơn, mở rộng dữ liệu, cải thiện mô hình và triển khai ứng dụng lên nền tảng cloud.
