import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import psutil  # Library for monitoring system resources

# Define memory usage threshold (in MB)
MEMORY_THRESHOLD_MB = 900  # Set the limit at which to clear cache and reload data

def clear_cache_if_memory_high():
    # Get the current memory usage
    memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB
    
    # Check if memory usage exceeds the threshold
    if memory_usage > MEMORY_THRESHOLD_MB:
        st.cache_data.clear()      # Clear cached data
        st.cache_resource.clear()  # Clear cached resources
        st.warning("Memory usage is high. Cache has been cleared to free up resources. Reloading data...")

        # Reload data and model after clearing cache
        global loaded_lstm_model, df_ticker, df_info
        loaded_lstm_model = load_lstm_model()
        df_ticker, df_info = download_data()

# Load data from GitHub with cache expiration
@st.cache_data(ttl=600)  # Cache will auto-clear after 10 minutes
def download_data():
    url_ticker = "https://raw.githubusercontent.com/letrongnghia1203/Streamlit-app/refs/heads/main/VN2023-data.xlsx%20-%20Ticker.csv"
    url_info = "https://raw.githubusercontent.com/letrongnghia1203/Streamlit-app/refs/heads/main/VN2023-data.xlsx%20-%20Info.csv"
    
    # Read data directly from GitHub
    df_ticker = pd.read_csv(url_ticker, low_memory=False)
    df_info = pd.read_csv(url_info, low_memory=False)
    return df_ticker, df_info

# Load LSTM model with caching
@st.cache_resource(ttl=600)  # Cache will auto-clear after 10 minutes
def load_lstm_model():
    # Load the model from a local file
    return load_model('best_lstm_model.keras')

# Run memory check before loading resources
clear_cache_if_memory_high()

# Load data and model
loaded_lstm_model = load_lstm_model()
df_ticker, df_info = download_data()

# Data preprocessing
df_info.columns = [col.replace('.', '_') for col in df_info.columns]
df_joined = pd.merge(df_info, df_ticker, on="Name", how="inner")
ticker_date_columns = [col for col in df_ticker.columns if '/' in col]

# Convert data from wide format to long format
df_vietnam = df_joined.melt(id_vars=list(df_info.columns) + ["Code"],
                            value_vars=ticker_date_columns,
                            var_name="Ngày", value_name="Giá đóng cửa")

df_vietnam["Symbol"] = df_vietnam["Symbol"].str.replace("^VT:", "", regex=True)
df_vietnam["Ngày"] = pd.to_datetime(df_vietnam["Ngày"], format='%m/%d/%Y', errors='coerce')
df_vietnam["Giá đóng cửa"] = pd.to_numeric(df_vietnam["Giá đóng cửa"].str.replace(',', '.'), errors='coerce')
df_vietnam = df_vietnam.dropna(subset=["Giá đóng cửa"])
df_vietnam = df_vietnam[df_vietnam["Exchange"] != "Hanoi OTC"]

# Display app title
st.title("Stock Market Data Visualization with LSTM Predictions")

# Get user input for stock symbol
symbol = st.text_input("Nhập mã cổ phiếu để xem thông tin chi tiết và dự đoán:")

# Function to create sequences for LSTM
def create_sequences(data, seq_length=5):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

# If user inputs a stock symbol
if symbol:
    # Filter data by stock symbol
    df_filtered = df_vietnam[df_vietnam['Symbol'] == symbol.upper()]

    if not df_filtered.empty:
        # Display one row of stock information
        st.write(f"Thông tin chi tiết của mã cổ phiếu {symbol.upper()}:")
        st.write(df_filtered.head(1))  # Show only one row

        # Create Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered['Ngày'], y=df_filtered['Giá đóng cửa'], mode='lines+markers', name='Giá Đóng Cửa'))
        
        # Prepare data for LSTM prediction
        prices = df_filtered[['Giá đóng cửa']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices)

        # Generate sequences for LSTM
        seq_length = 5
        X, _ = create_sequences(prices_scaled, seq_length)

        if len(X) > 0:
            # Predict with LSTM
            predictions = loaded_lstm_model.predict(X)
            predictions = scaler.inverse_transform(predictions)

            # Add predictions to chart
            prediction_dates = df_filtered['Ngày'].iloc[seq_length:].values
            fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Dự đoán'))

        fig.update_layout(title=f'Giá Đóng Cửa Cổ Phiếu {symbol.upper()} với Dự Đoán LSTM',
                          xaxis_title='Ngày', yaxis_title='Giá Đóng Cửa (VND)', template='plotly_white')
        
        # Display chart on Streamlit
        st.plotly_chart(fig)
    else:
        st.write("Không có dữ liệu cho mã cổ phiếu này.")
else:
    st.write("Vui lòng nhập mã cổ phiếu để xem thông tin chi tiết.")
