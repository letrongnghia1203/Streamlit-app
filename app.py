import streamlit as st
import gdown
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Tải dữ liệu từ Google Drive
gdown.download("https://drive.google.com/uc?export=download&id=1x1CkrJRe6PTOdWouYLhqG3f8MEXP-kbl", "VN2023-data-Ticker.csv", quiet=False)
gdown.download("https://drive.google.com/uc?export=download&id=1M9GA96Zhoj9HzqMPIlfnMeK7pob1bv2z", "VN2023-data-Info.csv", quiet=False)
# Tải mô hình từ Google Drive
model_id = '1-2diAZCXfnoe38o21Vv5Sx8wmre1IceY'
gdown.download(f'https://drive.google.com/uc?export=download&id={model_id}', 'best_lstm_model.keras', quiet=False)

# Load mô hình
loaded_lstm_model = load_model('best_lstm_model.keras')

# Đọc dữ liệu bằng Pandas
df_ticker = pd.read_csv("VN2023-data-Ticker.csv")
df_info = pd.read_csv("VN2023-data-Info.csv")

# Tiền xử lý dữ liệu
# Đổi tên cột để bỏ dấu chấm (nếu có)
df_info.columns = [col.replace('.', '_') for col in df_info.columns]

# Ghép nối bảng dựa trên cột "Name"
df_joined = pd.merge(df_info, df_ticker, on="Name", how="inner")

# Lấy các cột chứa ngày để xử lý
ticker_date_columns = [col for col in df_ticker.columns if '/' in col]

# Chuyển đổi dữ liệu từ dạng rộng sang dạng dài
df_vietnam = df_joined.melt(id_vars=list(df_info.columns) + ["Code"],
                            value_vars=ticker_date_columns,
                            var_name="Ngày", value_name="Giá đóng cửa")

# Xóa chuỗi "VT:" từ cột Symbol nếu có
df_vietnam["Symbol"] = df_vietnam["Symbol"].str.replace("^VT:", "", regex=True)

# Chuyển đổi cột "Ngày" thành kiểu ngày tháng và cột "Giá đóng cửa" thành số float
df_vietnam["Ngày"] = pd.to_datetime(df_vietnam["Ngày"], format='%m/%d/%Y', errors='coerce')
df_vietnam["Giá đóng cửa"] = pd.to_numeric(df_vietnam["Giá đóng cửa"].str.replace(',', '.'), errors='coerce')

# Lọc bỏ các dòng có giá trị null và những dòng thuộc "Hanoi OTC"
df_vietnam = df_vietnam.dropna(subset=["Giá đóng cửa"])
df_vietnam = df_vietnam[df_vietnam["Exchange"] != "Hanoi OTC"]

# Hiển thị dữ liệu trong Streamlit
st.title("Stock Market Data Visualization with LSTM Predictions")
st.write("Dữ liệu sau khi xử lý:")
st.write(df_vietnam.head(30))

# Lấy input mã cổ phiếu từ người dùng
symbol = st.text_input("Nhập mã cổ phiếu để xem biểu đồ và dự đoán:")

# Hàm tạo chuỗi cho LSTM
def create_sequences(data, seq_length=5):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

# Hiển thị biểu đồ và dự đoán nếu có mã cổ phiếu
if symbol:
    # Lọc dữ liệu theo mã cổ phiếu
    df_filtered = df_vietnam[df_vietnam['Symbol'] == symbol.upper()]

    if not df_filtered.empty:
        # Tạo biểu đồ với Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered['Ngày'], y=df_filtered['Giá đóng cửa'], mode='lines+markers', name='Giá Đóng Cửa'))
        
        # Chuẩn bị dữ liệu cho dự đoán LSTM
        prices = df_filtered[['Giá đóng cửa']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices)

        # Chuỗi dữ liệu cho LSTM
        seq_length = 5
        X, _ = create_sequences(prices_scaled, seq_length)

        if len(X) > 0:
            # Dự đoán bằng LSTM
            predictions = loaded_lstm_model.predict(X)
            predictions = scaler.inverse_transform(predictions)

            # Thêm dự đoán vào biểu đồ
            prediction_dates = df_filtered['Ngày'].iloc[seq_length:].values
            fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Dự đoán'))

        fig.update_layout(title=f'Giá Đóng Cửa Cổ Phiếu {symbol.upper()} với Dự Đoán LSTM',
                          xaxis_title='Ngày', yaxis_title='Giá Đóng Cửa (VND)', template='plotly_white')
        
        # Hiển thị biểu đồ trên Streamlit
        st.plotly_chart(fig)
    else:
        st.write("Không có dữ liệu cho mã cổ phiếu này.")
