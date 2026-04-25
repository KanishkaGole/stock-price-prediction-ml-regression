import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import math

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ADANIPORTS.csv")
    df.dropna(inplace=True)
    return df

# App title
st.set_page_config(layout="wide")
df = load_data()
st.title("📈 AdaniPorts Stock Price Prediction")

# Time Series Visualization
st.subheader("📊 Time Series Plot of Closing Price")
fig, ax = plt.subplots(figsize=(10, 4))
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Close'].plot(ax=ax, label='Close Price')
ax.set_title("Close Price Over Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Sidebar - Model selection
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR()
}
st.sidebar.header("Model Options")
model_choice = st.sidebar.selectbox("Select a model", list(models.keys()))

# Feature Engineering
df['Daily Change'] = df['Close'] - df['Open']
df['Change %'] = ((df['Close'] - df['Open']) / df['Open']) * 100
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['STD_5'] = df['Close'].rolling(window=5).std()
df.dropna(inplace=True)

features = ['Open', 'Volume', 'Daily Change', 'Change %', 'MA_5', 'MA_10', 'STD_5']
target = 'Close'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = models[model_choice]
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "📌 Evaluation", "📉 Comparison", "📅 Forecast"])

with tab1:
    st.subheader("Feature Correlation")
    corr = df[features + [target]].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(plt)

with tab2:
    st.subheader(f"Evaluation: {model_choice}")
    st.metric("R² Score", f"{r2:.4f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")
    fig, ax = plt.subplots()
    ax.plot(y_test.values[:50], label='Actual')
    ax.plot(predictions[:50], label='Predicted', linestyle='--')
    ax.set_title("Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.subheader("Model Comparison")
    results = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        results.append({
            "Model": name,
            "R2 Score": r2_score(y_test, p),
            "MAE": mean_absolute_error(y_test, p),
            "RMSE": math.sqrt(mean_squared_error(y_test, p))
        })
    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    st.dataframe(results_df)
    st.bar_chart(results_df.set_index("Model")[["R2 Score", "RMSE"]])


with tab4:
    st.subheader("🔮 Forecast Future Prices")
    forecast_model = models[model_choice]
    forecast_model.fit(X, y)

    days_ahead = st.slider("Select # future days", 1, 60, 14)
    future_dates = []
    future_preds = []

    # Start from last known values
    recent_df = df[-10:].copy()
    forecast_df = pd.DataFrame()

    for i in range(1, days_ahead + 1):
        next_date = df.index[-1] + pd.Timedelta(days=i)
        future_dates.append(next_date)

        # Simulate next open as previous close
        next_open = recent_df['Close'].iloc[-1]

        # Volume estimate using small random noise on last value
        next_volume = recent_df['Volume'].iloc[-1] * np.random.uniform(0.98, 1.02)

        # Close placeholder for now
        temp_close = next_open

        daily_change = temp_close - next_open
        change_pct = (daily_change / next_open) * 100
        ma_5 = recent_df['Close'].rolling(window=5).mean().iloc[-1]
        ma_10 = recent_df['Close'].rolling(window=10).mean().iloc[-1]
        std_5 = recent_df['Close'].rolling(window=5).std().iloc[-1]

        feature_row = np.array([[next_open, next_volume, daily_change, change_pct, ma_5, ma_10, std_5]])
        predicted_close = forecast_model.predict(feature_row)[0]
        future_preds.append(predicted_close)

        # Append new row to recent_df for next loop's rolling features
        next_row = pd.DataFrame({
            'Open': [next_open],
            'Close': [predicted_close],
            'Volume': [next_volume]
        }, index=[next_date])
        recent_df = pd.concat([recent_df, next_row])
        if len(recent_df) > 10:
            recent_df = recent_df[-10:]

    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds})
    forecast_df.set_index('Date', inplace=True)

    # Combine historical and forecast data for visualization
    combined_df = pd.concat([
        df[['Close']].rename(columns={'Close': 'Price'}),
        forecast_df.rename(columns={'Predicted Close': 'Price'})
    ])

    # Plot time series with both actual and forecast
    st.subheader("📈 Combined Time Series: Actual + Forecast")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    combined_df['Price'].plot(ax=ax2, label="Historical", color='blue')
    forecast_df['Predicted Close'].plot(ax=ax2, label="Forecast", color='orange')
    ax2.set_title("AdaniPorts Stock Price Forecast")
    ax2.set_ylabel("Price (₹)")
    ax2.legend()
    st.pyplot(fig2)


    st.subheader("📋 Forecast Table")
    st.dataframe(forecast_df.style.format({"Predicted Close": "₹{:.2f}"}))

    # Specific date prediction
    st.subheader("🔍 Predict on Specific Future Date")
    min_d = forecast_df.index.min().date()
    max_d = forecast_df.index.max().date()
    selected_d = st.date_input("Select Date", min_value=min_d, max_value=max_d, value=min_d)
    if pd.to_datetime(selected_d) in forecast_df.index:
        price = forecast_df.loc[pd.to_datetime(selected_d), "Predicted Close"]
        st.success(f"Predicted Close Price on {selected_d}: ₹{price:.2f}")
    else:
        st.warning("Select a date within range.")

    # Download forecast
    csv_data = forecast_df.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast", data=csv_data, file_name="forecast.csv", mime='text/csv')


# Download model predictions
st.download_button(
    "📥 Download Model Predictions",
    data=pd.DataFrame({"Actual": y_test.values, "Predicted": predictions}).to_csv(index=False),
    file_name="model_predictions.csv",
    mime="text/csv"
)
