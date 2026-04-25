# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import math

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ADANIPORTS.csv")
    df.dropna(inplace=True)
    return df

# App config
st.set_page_config(layout="wide")
st.title("📈 AdaniPorts Stock Price Prediction")

# Load and prepare data
df = load_data()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Time Series Chart
st.subheader("📊 Time Series Plot of Closing Price")
fig, ax = plt.subplots(figsize=(10, 4))
df['Close'].plot(ax=ax, label='Close Price')
ax.set_title("Close Price Over Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Sidebar Model Selection
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR(),
    "AdaBoost Regressor": AdaBoostRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "LightGBM Regressor": lgb.LGBMRegressor(n_estimators=100, random_state=42)
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

# Model training and evaluation
model = models[model_choice]
model.fit(X_train, y_train)
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = math.sqrt(mean_squared_error(y_test, predictions))

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
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test.values[:50], label='Actual Close')
    ax.plot(predictions[:50], label='Predicted Close', linestyle='--')

    # Predict Open for same test set
    open_y = df['Open']
    _, X_test_open, _, y_test_open = train_test_split(X, open_y, test_size=0.2, random_state=42)
    pred_open = model.predict(X_test_open)

    ax.plot(y_test_open.values[:50], label='Actual Open')
    ax.plot(pred_open[:50], label='Predicted Open', linestyle='--')

    ax.set_title("Actual vs Predicted - Close & Open")
    ax.legend()
    st.pyplot(fig)


with tab3:
    st.subheader("Model Comparison")
    results = []
    open_y = df['Open']
    _, X_test_open, _, y_test_open = train_test_split(X, open_y, test_size=0.2, random_state=42)

    for name, m in models.items():
        m.fit(X_train, y_train)
        p_close = m.predict(X_test)

        m.fit(X_train, open_y.loc[X_train.index])
        p_open = m.predict(X_test_open)

        results.append({
            "Model": name,
            "R2 Close": r2_score(y_test, p_close),
            "RMSE Close": math.sqrt(mean_squared_error(y_test, p_close)),
            "R2 Open": r2_score(y_test_open, p_open),
            "RMSE Open": math.sqrt(mean_squared_error(y_test_open, p_open)),
        })

    results_df = pd.DataFrame(results).sort_values(by="R2 Close", ascending=False)
    st.dataframe(results_df)

    st.bar_chart(results_df.set_index("Model")[["R2 Close", "R2 Open"]])



with tab4:
    st.subheader("🔮 Forecast Future Open & Close Prices")

    # Train on full dataset for both Open and Close
    forecast_model_close = models[model_choice]
    forecast_model_open = models[model_choice]

    forecast_model_close.fit(X, y)
    forecast_model_open.fit(X, df['Open'])

    days_ahead = st.slider("Select number of future days", 1, 365, 30)
    forecast_df = pd.DataFrame(columns=['Date', 'Predicted Open', 'Predicted Close'])

    last_close_vals = list(df['Close'].values[-10:])
    last_open_vals = list(df['Open'].values[-10:])
    last_volume = df['Volume'].iloc[-1]

    for i in range(1, days_ahead + 1):
        date = df.index[-1] + pd.Timedelta(days=i)

        next_open = last_close_vals[-1]  # Approximate next open as last close
        daily_change = last_close_vals[-1] - next_open
        change_pct = (daily_change / next_open) * 100
        ma_5 = np.mean(last_close_vals[-5:])
        ma_10 = np.mean(last_close_vals)
        std_5 = np.std(last_close_vals[-5:])
        features_future = np.array([[next_open, last_volume, daily_change, change_pct, ma_5, ma_10, std_5]])

        pred_close = forecast_model_close.predict(features_future)[0]
        pred_open = forecast_model_open.predict(features_future)[0]

        last_close_vals.append(pred_close)
        last_open_vals.append(pred_open)

        forecast_df = pd.concat([
            forecast_df,
            pd.DataFrame([[date, pred_open, pred_close]], columns=['Date', 'Predicted Open', 'Predicted Close'])
        ], ignore_index=True)

    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_df.set_index('Date', inplace=True)

    # Combined chart: actual + predicted Close price
    combined_df = pd.concat([df[['Close']], forecast_df[['Predicted Close']].rename(columns={'Predicted Close': 'Close'})])

    st.subheader("📈 Combined Time Series: Actual + Forecast")
    range_option = st.selectbox("Show data from:", ["Last 6 Months", "Last 1 Year", "Last 3 Years", "Last 5 Years", "All"])
    if range_option == "Last 6 Months":
        plot_start_date = df.index.max() - pd.DateOffset(months=6)
    elif range_option == "Last 1 Year":
        plot_start_date = df.index.max() - pd.DateOffset(years=1)
    elif range_option == "Last 3 Years":
        plot_start_date = df.index.max() - pd.DateOffset(years=3)
    elif range_option == "Last 5 Years":
        plot_start_date = df.index.max() - pd.DateOffset(years=5)
    else:
        plot_start_date = df.index.min()

        # Prepare combined DataFrame
    actual_trimmed = df[df.index >= plot_start_date][['Open', 'Close']].copy()
    forecast_trimmed = forecast_df.copy()

    actual_trimmed.rename(columns={"Open": "Actual Open", "Close": "Actual Close"}, inplace=True)
    forecast_trimmed.rename(columns={
        "Predicted Open": "Forecast Open",
        "Predicted Close": "Forecast Close"
    }, inplace=True)

    combined_chart_df = pd.concat([actual_trimmed, forecast_trimmed], axis=0)
    combined_chart_df = combined_chart_df.sort_index()

    st.line_chart(combined_chart_df)


    # Forecast Table with Open & Close
    st.subheader("📋 Forecast Table")
    st.dataframe(forecast_df.style.format({"Predicted Open": "₹{:.2f}", "Predicted Close": "₹{:.2f}"}))

    # Predict on specific date
    st.subheader("🔍 Predict on Specific Future Date")
    min_d = forecast_df.index.min().date()
    max_d = forecast_df.index.max().date()
    selected_d = st.date_input("Select Date", min_value=min_d, max_value=max_d, value=min_d)
    if pd.to_datetime(selected_d) in forecast_df.index:
        open_val = forecast_df.loc[pd.to_datetime(selected_d), "Predicted Open"]
        close_val = forecast_df.loc[pd.to_datetime(selected_d), "Predicted Close"]
        st.success(f"📅 {selected_d}: Predicted Open = ₹{open_val:.2f}, Predicted Close = ₹{close_val:.2f}")
    else:
        st.warning("Select a date within the forecast range.")

    # Download
    csv_data = forecast_df.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast (Open & Close)", data=csv_data, file_name="forecast_open_close.csv", mime='text/csv')


# Download model predictions
st.download_button(
    "📥 Download Model Predictions",
    data=pd.DataFrame({"Actual": y_test.values, "Predicted": predictions}).to_csv(index=False),
    file_name="model_predictions.csv",
    mime="text/csv"
)
