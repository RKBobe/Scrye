# app.py - Your Crypto Prediction Web Application

# --- 1. Imports ---
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import pandas_ta as ta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- 2. Helper Functions (Our existing pipeline) ---

def get_top_coins(limit=20):
    """Fetches the top cryptocurrencies by market cap from CoinGecko."""
    print("Fetching top 20 coins...")
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': limit, 'page': 1}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        # Format for yfinance (e.g., 'BTC-USD')
        return [coin['symbol'].upper() + '-USD' for coin in response.json()]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching top coins: {e}")
        # Return a default list if the API fails
        return ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD']

def load_and_clean_coin(ticker_symbol):
    """Loads and cleans data for a given crypto ticker."""
    file_path = os.path.join('data', f'{ticker_symbol}_data.csv')
    if not os.path.exists(file_path):
        print(f"Data for {ticker_symbol} not found. Please run get_data.py first.")
        return None
        
    col_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    try:
        df = pd.read_csv(
            file_path, header=None, skiprows=3, names=col_names,
            index_col='Date', parse_dates=True
        )
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

def create_features(df):
    """Creates a comprehensive set of features for modeling."""
    if df is None: return None
    df_featured = df.copy()
    for i in range(1, 8):
        df_featured[f'Close_lag_{i}'] = df_featured['Close'].shift(i)
    df_featured['MA_7'] = df_featured['Close'].rolling(window=7).mean()
    df_featured['MA_30'] = df_featured['Close'].rolling(window=30).mean()
    df_featured['Volatility_7'] = df_featured['Close'].rolling(window=7).std()
    df_featured['RSI_14'] = ta.rsi(df_featured['Close'], length=14)
    df_featured['Target'] = df_featured['Close'].shift(-1)
    return df_featured

# --- 3. The Main Prediction and Visualization Function ---

def predict_and_visualize(ticker_symbol):
    """
    This master function runs the entire pipeline for a given ticker
    and returns the prediction and the plot.
    """
    print(f"--- Running pipeline for {ticker_symbol} ---")
    
    # Step 1: Load and prepare data
    df = load_and_clean_coin(ticker_symbol)
    if df is None:
        return "Could not load data for this coin. Please ensure data exists in the /data folder.", None
    
    featured_df = create_features(df)
    
    # Step 2: Prepare data for modeling
    feature_cols = [col for col in featured_df.columns if 'lag' in col or 'MA' in col or 'Volatility' in col or 'RSI' in col]
    model_df = featured_df[feature_cols + ['Target']].copy()
    model_df.dropna(inplace=True)
    
    if len(model_df) < 20: # Ensure there's enough data to train
        return "Not enough historical data to make a prediction for this coin.", None

    X = model_df[feature_cols]
    y = model_df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Step 3: Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Step 4: Make prediction for tomorrow
    latest_features = model_df[feature_cols].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(latest_features)[0]
    prediction_text = f"The predicted closing price for {ticker_symbol} tomorrow is: ${prediction:,.2f}"
    
    # Step 5: Create visualization
    test_predictions = model.predict(X_test)
    results = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': test_predictions}, index=y_test.index)
    
    fig = plt.figure(figsize=(12, 6))
    plt.plot(results['Actual Price'], label='Actual Price', color='blue')
    plt.plot(results['Predicted Price'], label='Model Prediction', color='red', linestyle='--')
    plt.title(f'{ticker_symbol}: Actual vs. Predicted Price on Test Data', fontsize=16)
    plt.legend()
    plt.grid(True)
    
    # Close the plot to prevent it from displaying directly in the console
    plt.close(fig)
    
    return prediction_text, fig

# --- 4. Build and Launch the Gradio Interface ---

# Fetch the list of coins for the dropdown
top_20_coins = get_top_coins()

# Create the Gradio Interface
iface = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Dropdown(top_20_coins, label="Select a Cryptocurrency"),
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Plot(label="Prediction Chart")
    ],
    title="Cryptocurrency Price Predictor",
    description="Select one of the top 20 cryptocurrencies to get a price prediction for the next day. Based on a Linear Regression model with lag, moving average, volatility, and RSI features.",
    allow_flagging="never"
)

# Launch the app!
print("Launching Gradio App...")
iface.launch(server_name="0.0.0.0")