# Crypto Price Prediction Model

This Codespace is configured for developing a cryptocurrency price prediction model using Python.  
It includes the following:

- Python 3.10
- Data science libraries (NumPy, Pandas, scikit-learn, Matplotlib)
- Jupyter Notebook
- yfinance for data collection

## Getting Started

1. Launch this Codespace.
2. All dependencies will be installed automatically.
3. Place your datasets in the `/data` directory.
4. Start Jupyter Notebook with:

    ```bash
    jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
    ```

## Ports

- 8888: Jupyter Notebook
- 5000: Flask (if you deploy an API)

## Customization

Edit `.devcontainer/devcontainer.json` for further tweaks.

TODO: 
This is a comprehensive plan to expand the current Scrye application from a functional prototype into a robust, production-ready financial prediction tool.

The plan is structured into four phases, focusing on data integrity, advanced modeling, system robustness, and enhanced user experience.

Expansion Plan: Scrye - Production-Ready Financial Tool
Phase 1: Data Integrity and Visualization Foundation (3-4 Weeks)
The objective of this phase is to eliminate reliance on static files, ensure data freshness, and upgrade the core visualization capabilities.

Objective	Task	Details & Impact
1.1 Live Data Integration	Replace static file loading.	Modify load_and_clean_coin to use a reliable library (e.g., yfinance or a dedicated CoinGecko wrapper) to fetch the latest OHLCV data directly from the source upon request.
1.2 Candlestick Visualization	Implement a dedicated charting function.	Introduce mplfinance to generate professional candlestick charts alongside the existing line plot. This provides users with essential OHLC visual context.
1.3 Data Caching & Error Handling	Implement a caching layer.	Use functools.lru_cache or a simple file-based cache to store recently fetched data. This prevents excessive API calls and speeds up repeated requests. Add robust try...except blocks for API rate limits and connection failures.
1.4 Feature Expansion (Technical Indicators)	Add advanced indicators.	Integrate more complex indicators using pandas_ta, such as MACD, Bollinger Bands (BBANDS), and Average True Range (ATR), into the create_features function.
Phase 2: Advanced Modeling and Evaluation (6-8 Weeks)
The objective of this phase is to move beyond simple Linear Regression, introduce more powerful algorithms, and provide quantifiable proof of the model's performance.

Objective	Task	Details & Impact
2.1 Model Diversification	Integrate advanced ML models.	Add support for XGBoost Regressor and Random Forest Regressor. Create a mechanism (e.g., a dictionary mapping model names to classes) to allow the user to select the model via the Gradio interface.
2.2 Time-Series Specific Modeling	Introduce Deep Learning/Statistical models.	Implement a basic LSTM (Long Short-Term Memory) model for sequence prediction, or a statistical model like ARIMA/SARIMA. This is crucial for capturing complex time-series dependencies.
2.3 Comprehensive Evaluation Metrics	Calculate and display performance.	In the predict_and_visualize function, calculate and return key metrics for the test set: RMSE, MAE, R-squared, and Directional Accuracy (percentage of correct up/down predictions).
2.4 Hyperparameter Tuning	Implement a tuning module.	Add an optional step using GridSearchCV or RandomizedSearchCV (for the non-deep learning models) to optimize model parameters, improving prediction accuracy.
2.5 Prediction Confidence	Calculate and visualize confidence intervals.	Instead of just a single point prediction, use techniques (like bootstrapping or model variance) to calculate a 95% confidence interval and display this range on the prediction chart.
Phase 3: Production Readiness and System Robustness (4-6 Weeks)
The objective of this phase is to prepare the application for reliable, scalable deployment in a production environment.

Objective	Task	Details & Impact
3.1 Containerization	Create a Dockerfile and Docker Compose.	Containerize the application and its dependencies. This ensures the application runs identically across all environments (development, testing, production).
3.2 Structured Logging	Implement a logging system.	Use Python's built-in logging module to track key events: data fetching errors, model training times, prediction results, and user interactions. Log to a file or standard output.
3.3 Database Integration (Optional but Recommended)	Set up a lightweight database (SQLite/PostgreSQL).	Store historical predictions, model performance metrics, and backtesting results. This allows users to review past forecasts and track model drift over time.
3.4 Code Refactoring and Modularity	Separate concerns into modules.	Break app.py into logical files: data_loader.py, feature_engineering.py, models.py, and interface.py. This improves maintainability and testability.
Phase 4: Advanced UX and Financial Utility (4-6 Weeks)
The objective of this phase is to transform the Gradio interface into a powerful dashboard and add core financial analysis features.

Objective	Task	Details & Impact
4.1 Gradio Dashboard Layout	Switch to gr.Blocks.	Replace the simple gr.Interface with gr.Blocks to create a multi-tab layout (e.g., "Prediction," "Backtesting," "Model Performance").
4.2 Backtesting Module	Allow historical simulation.	Create a dedicated function where users can select a historical date range and run the model against that period. Display simulated P&L (Profit and Loss) and key trading metrics (e.g., Sharpe Ratio, Max Drawdown).
4.3 User Customization	Add input sliders/fields.	Allow users to customize key parameters: Forecast Horizon (1 to 7 days), Train/Test Split Ratio, and Feature Lookback Periods (e.g., length of MA).
4.4 Risk Assessment Display	Visualize volatility and risk.	Display the current 7-day volatility (already calculated in create_features) and provide a simple risk score based on ATR or historical price swings.
4.5 Automated Retraining	Implement a scheduled retraining mechanism.	If deployed on a server, set up a cron job or a background process to automatically retrain the selected model daily using the latest data, ensuring the predictions remain relevant.
