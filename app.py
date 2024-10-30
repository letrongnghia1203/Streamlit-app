import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Hàm khởi tạo mô hình LSTM mới
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Khởi tạo mô hình LSTM mà không cần tải từ tệp
lstm_model = create_lstm_model((5, 1))  # Thiết lập hình dạng đầu vào của bạn

# Tải dữ liệu từ Google Drive
@st.cache_data
def download_data():
    url = "https://drive.google.com/uc?export=download&id=1x1CkrJRe6PTOdWouYLhqG3f8MEXP-kbl"
    df_ticker = pd.read_csv(url)
    url_info = "https://drive.google.com/uc?export=download&id=1M9GA96Zhoj9HzqMPIlfnMeK7pob1bv2z"
    df_info = pd.read_csv(url_info)
    return df_ticker, df_info

df_ticker, df_info = download_data()

# Xử lý dữ liệu
df_info.columns = [col.replace('.', '_') for col in df_info.columns]
df_joined = pd.merge(df_info, df_ticker, on="Name", how="inner")
ticker_date_columns = [col for col in df_ticker.columns if '/' in col]

# Chuyển đổi dữ liệu từ dạng rộng sang dạng dài
df_vietnam = df_joined.melt(id_vars=list(df_info.columns) + ["Code"],
                            value_vars=ticker_date_columns,
                            var_name="Ngày", value_name="Giá đóng cửa")

df_vietnam["Symbol"] = df_vietnam["Symbol"].str.replace("^VT:", "", regex=True)
df_vietnam["Ngày"] = pd.to_datetime(df_vietnam["Ngày"], format='%m/%d/%Y', errors='coerce')
df_vietnam["Giá đóng cửa"] = pd.to_numeric(df_vietnam["Giá đóng cửa"].str.replace(',', '.'), errors='coerce')
df_vietnam = df_vietnam.dropna(subset=["Giá đóng cửa"])
df_vietnam = df_vietnam[df_vietnam["Exchange"] != "Hanoi OTC"]

# Hiển thị tiêu đề ứng dụng
st.title("Stock Market Data Visualization with LSTM Predictions")

# Lấy input mã cổ phiếu từ người dùng
symbol = st.text_input("Nhập mã cổ phiếu để xem thông tin chi tiết và dự đoán:")

# Hàm tạo chuỗi cho LSTM
def create_sequences(data, seq_length=5):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

# Nếu người dùng nhập mã cổ phiếu
if symbol:
    # Lọc dữ liệu theo mã cổ phiếu
    df_filtered = df_vietnam[df_vietnam['Symbol'] == symbol.upper()]

    if not df_filtered.empty:
        # Hiển thị một dòng thông tin của mã cổ phiếu
        st.write(f"Thông tin chi tiết của mã cổ phiếu {symbol.upper()}:")
        st.write(df_filtered.head(1))  # Hiển thị chỉ một dòng

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
            # Dự đoán bằng mô hình LSTM (chưa được huấn luyện nên kết quả chỉ là ví dụ)
            predictions = lstm_model.predict(X)
            predictions = scaler.inverse_transform(predictions)

            # Thêm dự đoán vào biểu đồ
            prediction_dates = df_filtered['Ngày'].iloc[seq_length:].values
            fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Dự đoán (untrained)'))

        fig.update_layout(title=f'Giá Đóng Cửa Cổ Phiếu {symbol.upper()} với Dự Đoán LSTM',
                          xaxis_title='Ngày', yaxis_title='Giá Đóng Cửa (VND)', template='plotly_white')
        
        # Hiển thị biểu đồ trên Streamlit
        st.plotly_chart(fig)
    else:
        st.write("Không có dữ liệu cho mã cổ phiếu này.")
else:
    st.write("Vui lòng nhập mã cổ phiếu để xem thông tin chi tiết.")
