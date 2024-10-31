import streamlit as st
import gdown
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load data only once and store in session state
@st.cache_resource
def load_data():
    gdown.download("https://drive.google.com/uc?export=download&id=1x1CkrJRe6PTOdWouYLhqG3f8MEXP-kbl", "VN2023-data-Ticker.csv", quiet=False)
    gdown.download("https://drive.google.com/uc?export=download&id=1M9GA96Zhoj9HzqMPIlfnMeK7pob1bv2z", "VN2023-data-Info.csv", quiet=False)
    df_ticker = pd.read_csv("VN2023-data-Ticker.csv", low_memory=False)
    df_info = pd.read_csv("VN2023-data-Info.csv", low_memory=False)
    df_info.columns = [col.replace('.', '_') for col in df_info.columns]
    df_joined = pd.merge(df_info, df_ticker, on="Name", how="inner")
    ticker_date_columns = [col for col in df_ticker.columns if '/' in col]

    df_vietnam = df_joined.melt(
        id_vars=list(df_info.columns) + ["Code"],
        value_vars=ticker_date_columns,
        var_name="Ngày", value_name="Giá đóng cửa"
    )
    df_vietnam["Symbol"] = df_vietnam["Symbol"].str.replace("^VT:", "", regex=True)
    df_vietnam["Ngày"] = pd.to_datetime(df_vietnam["Ngày"], format='%m/%d/%Y', errors='coerce')
    df_vietnam["Giá đóng cửa"] = pd.to_numeric(df_vietnam["Giá đóng cửa"].str.replace(',', '.'), errors='coerce')
    return df_vietnam.dropna(subset=["Giá đóng cửa"])

# Load LSTM model only once
@st.cache_resource
def load_lstm_model():
    model_id = '1-2diAZCXfnoe38o21Vv5Sx8wmre1IceY'
    gdown.download(f'https://drive.google.com/uc?export=download&id={model_id}', 'best_lstm_model.keras', quiet=False)
    return load_model('best_lstm_model.keras')

# Initialize data and model in session state if not present
if 'df_vietnam' not in st.session_state:
    st.session_state.df_vietnam = load_data()

if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = load_lstm_model()

# Display title
st.title("Stock Market Data Visualization with LSTM Predictions")

# Symbol input and old/new check
symbol = st.text_input("Nhập mã cổ phiếu để xem thông tin chi tiết và dự đoán:")
if symbol:
    df_filtered = st.session_state.df_vietnam[st.session_state.df_vietnam['Symbol'] == symbol.upper()]

    if not df_filtered.empty:
        st.write(f"Thông tin chi tiết của mã cổ phiếu {symbol.upper()}:")
        st.write(df_filtered.head(1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered['Ngày'], y=df_filtered['Giá đóng cửa'], mode='lines+markers', name='Giá Đóng Cửa'))
        
        # Prepare and scale data for prediction
        prices = df_filtered[['Giá đóng cửa']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices)

        # Create sequences for LSTM
        seq_length = 5
        X = np.array([prices_scaled[i:i + seq_length] for i in range(len(prices_scaled) - seq_length)])

        if len(X) > 0:
            # Predict and invert scaling
            predictions = st.session_state.lstm_model.predict(X)
            predictions = scaler.inverse_transform(predictions)
            prediction_dates = df_filtered['Ngày'].iloc[seq_length:].values

            fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Dự đoán'))

        fig.update_layout(title=f'Giá Đóng Cửa Cổ Phiếu {symbol.upper()} với Dự Đoán LSTM',
                          xaxis_title='Ngày', yaxis_title='Giá Đóng Cửa (VND)', template='plotly_white')
        
        st.plotly_chart(fig)
    else:
        st.write("Không có dữ liệu cho mã cổ phiếu này.")
else:
    st.write("Vui lòng nhập mã cổ phiếu để xem thông tin chi tiết.")
