# Weather Chatbot with Interactive Map + ML Dashboard

Ứng dụng Streamlit kết hợp:

- Chatbot hỏi thời tiết hiện tại
- Bản đồ thời tiết nhiều lớp
- Dự đoán nhiệt độ bằng Machine Learning
- Dashboard phân tích dữ liệu từ Kaggle

## Tính năng chính

### 1. Chatbot & thời tiết hiện tại
Người dùng có thể nhập:
- tên thành phố
- hoặc câu như: `thời tiết ở Hà Nội`, `weather in Da Nang`

Ứng dụng hiển thị:
- nhiệt độ
- độ ẩm
- áp suất
- tốc độ gió
- mô tả thời tiết

### 2. Bản đồ thời tiết nhiều lớp
Bản đồ có thể bật/tắt các lớp:
- Bản đồ nhiệt độ
- Khu vực đang có mưa
- Hướng và cường độ gió
- Mật độ mây
- Áp suất khí quyển

### 3. Machine Learning Prediction
Mô hình được huấn luyện từ dataset Kaggle để ước lượng nhiệt độ dựa trên dữ liệu dự báo.

Các mô hình được so sánh:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Các chỉ số đánh giá:
- MAE
- RMSE
- R²

### 4. Data Science Dashboard
Dashboard gồm:
- Bảng dữ liệu đã làm sạch
- Histogram nhiệt độ
- Histogram độ ẩm
- Biểu đồ xu hướng nhiệt độ
- Ma trận tương quan

## File trong project

- `app.py` → giao diện Streamlit
- `weather_ml.py` → xử lý dữ liệu dự báo và dự đoán ML
- `train_model.py` → huấn luyện mô hình từ dataset Kaggle
- `weatherHistory.csv` → dataset gốc
- `weather_dataset_cleaned.csv` → dataset đã làm sạch
- `model_comparison.csv` → kết quả so sánh mô hình
- `weather_model.pkl` → mô hình tốt nhất

## Cài đặt

```bash
pip install -r requirements.txt
streamlit run app.py
