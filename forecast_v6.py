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
import matplotlib.dates as mdates
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load dataset and exclude COVID years (2020-2021)
@st.cache_data
def load_data():
    df = pd.read_csv("ADANIPORTS.csv")
    # Specify dayfirst=True to handle DD-MM-YYYY format
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    # Exclude data from 2020 and 2021
    df = df[~df['Date'].dt.year.isin([2020, 2021])]
    # Verify that no data from 2020 or 2021 remains
    if not df['Date'].dt.year.isin([2020, 2021]).any():
        st.write(f"Data loaded successfully. Date range: {df['Date'].min().date()} to {df['Date'].max().date()}. No data from 2020 or 2021 included.")
    else:
        st.warning("Warning: Some data from 2020 or 2021 may still be present.")
    df.dropna(inplace=True)
    return df

st.set_page_config(layout="wide")
df = load_data()
st.title("📈 AdaniPorts Stock Price Prediction")

# Convert and set index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Time Series Visualization
st.subheader("📊 Time Series Plot of Closing Price")
fig, ax = plt.subplots(figsize=(10, 4))
df['Close'].plot(ax=ax, label='Close Price')
ax.set_title("Close Price Over Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Sidebar - Model selection
models = {
    "Linear Regression": LinearRegression(),
    
    "Decision Tree": DecisionTreeRegressor(
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    ),
    
    "Support Vector Regressor": Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(
            kernel='rbf',
            C=1.5,
            epsilon=0.1,
            gamma='scale',
            cache_size=1000
        ))
    ]),
    
    "AdaBoost Regressor": AdaBoostRegressor(
        n_estimators=150,
        learning_rate=0.5,
        loss='linear',
        random_state=42
    ),
    
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    
    "KNN Regressor": Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(
            n_neighbors=7,
            weights='distance',
            algorithm='auto',
            leaf_size=30,
            metric='minkowski',
            p=2
        ))
    ]),
    
    "LightGBM Regressor": lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=10,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42
    )
}

st.sidebar.header("Model Options")
model_choice = st.sidebar.selectbox("Select a model", list(models.keys()))

# Modify the models dictionary with optimized default parameters
if model_choice != "Linear Regression":
    st.sidebar.subheader("Model Parameters")

    if model_choice == "Decision Tree":
        st.sidebar.markdown("""
        **Decision Tree Parameters Info:**
        - **Max Depth**: Maximum depth of the tree. Lower values prevent overfitting.
        - **Min Samples Split**: Minimum samples required to split a node. Higher values make the model more conservative.
        - **Min Samples Leaf**: Minimum samples required in a leaf node. Higher values prevent creating too specific rules.
        """)
        max_depth = st.sidebar.slider("Max Depth", 1, 50, value=8)
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, value=5)
        min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, value=2)
        
        models[model_choice] = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    elif model_choice == "Random Forest":
        st.sidebar.markdown("""
        **Random Forest Parameters Info:**
        - **Number of Trees**: More trees increase accuracy but also computation time.
        - **Max Depth**: Maximum depth of each tree. Controls model complexity.
        - **Min Samples Split**: Minimum samples needed to split a node. Controls overfitting.
        - **Min Samples Leaf**: Minimum samples in leaf nodes. Higher values create more general rules.
        """)
        n_estimators = st.sidebar.slider("Number of Trees", 50, 500, value=200)
        max_depth = st.sidebar.slider("Max Depth", 1, 50, value=15)
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, value=5)
        min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, value=2)
        
        models[model_choice] = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt',
            random_state=42
        )

    elif model_choice == "Support Vector Regressor":
        st.sidebar.markdown("""
        **SVR Parameters Info:**
        - **C (Regularization)**: Controls trade-off between smooth decision boundary and training accuracy.
        - **Epsilon**: Width of the insensitive region. Affects number of support vectors.
        - **Gamma**: Kernel coefficient. Higher values mean more complex decision boundaries.
        - **Kernel**: Type of kernel function. 'rbf' works well for non-linear relationships.
        """)
        C = st.sidebar.slider("C (Regularization)", 0.1, 10.0, value=1.5)
        epsilon = st.sidebar.slider("Epsilon", 0.01, 0.5, value=0.1)
        gamma = st.sidebar.select_slider("Gamma", options=['scale', 'auto'], value='scale')
        kernel = st.sidebar.selectbox("Kernel", ['rbf', 'linear', 'poly'], index=0)
        
        models[model_choice].named_steps['svr'].set_params(
            C=C,
            epsilon=epsilon,
            gamma=gamma,
            kernel=kernel
        )

    elif model_choice == "AdaBoost Regressor":
        st.sidebar.markdown("""
        **AdaBoost Parameters Info:**
        - **Number of Estimators**: Number of weak learners. More estimators can improve performance.
        - **Learning Rate**: How much each classifier contributes to the final prediction.
        - **Loss Function**: How the model updates weights. 'linear' is usually good for regression.
        """)
        n_estimators = st.sidebar.slider("Number of Estimators", 50, 300, value=150)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 2.0, value=0.5)
        loss = st.sidebar.selectbox("Loss Function", ['linear', 'square', 'exponential'], index=0)
        
        models[model_choice] = AdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            random_state=42
        )

    elif model_choice == "Gradient Boosting":
        st.sidebar.markdown("""
        **Gradient Boosting Parameters Info:**
        - **Number of Estimators**: Number of boosting stages. More can improve accuracy but may overfit.
        - **Learning Rate**: Shrinks contribution of each tree. Lower values need more trees.
        - **Max Depth**: Maximum depth of each tree. Controls complexity of each weak learner.
        - **Subsample Ratio**: Fraction of samples used for training each tree. Helps prevent overfitting.
        """)
        n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, value=200)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, value=0.1)
        max_depth = st.sidebar.slider("Max Depth", 1, 15, value=5)
        subsample = st.sidebar.slider("Subsample Ratio", 0.1, 1.0, value=0.8)
        
        models[model_choice] = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )

    elif model_choice == "KNN Regressor":
        st.sidebar.markdown("""
        **KNN Parameters Info:**
        - **Number of Neighbors**: Number of neighbors to use. Higher values smooth out predictions.
        - **Leaf Size**: Affects speed of queries. Higher values speed up large datasets.
        - **Distance Metric**: How to calculate distance between points.
        - **Weight Function**: How to weight neighbor contributions to predictions.
        """)
        n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, value=7)
        leaf_size = st.sidebar.slider("Leaf Size", 1, 50, value=30)
        metric = st.sidebar.selectbox("Distance Metric", ['minkowski', 'euclidean', 'manhattan'], index=0)
        weights = st.sidebar.selectbox("Weight Function", ['uniform', 'distance'], index=1)
        
        models[model_choice].named_steps['knn'].set_params(
            n_neighbors=n_neighbors,
            leaf_size=leaf_size,
            metric=metric,
            weights=weights
        )

    elif model_choice == "LightGBM Regressor":
        st.sidebar.markdown("""
        **LightGBM Parameters Info:**
        - **Number of Estimators**: Number of boosting rounds. More can improve accuracy.
        - **Learning Rate**: Controls impact of each tree on final outcome.
        - **Max Depth**: Maximum tree depth. -1 means no limit.
        - **Number of Leaves**: Maximum number of leaves in each tree.
        """)
        n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, value=200)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, value=0.1)
        max_depth = st.sidebar.slider("Max Depth", -1, 20, value=10)
        num_leaves = st.sidebar.slider("Number of Leaves", 2, 100, value=31)
        
        models[model_choice] = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42
        )

    # Add a general note about model complexity
    st.sidebar.markdown("""
    ---
    **Note**: Higher complexity (larger parameter values) may lead to:
    - Better fit on training data
    - Longer training time
    - Potential overfitting
    
    Lower complexity may lead to:
    - More generalization
    - Faster training
    - Potential underfitting
    
    Adjust parameters based on your data size and model performance.
    """)

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

# Add feature importance visualization for tree-based models
if model_choice in ["Decision Tree", "Random Forest", "Gradient Boosting", "LightGBM Regressor"]:
    with st.spinner("Training model and calculating feature importance..."):
        model = models[model_choice]
        model.fit(X_train, y_train)
        
        # Get feature importance
        if model_choice == "LightGBM Regressor":
            importance = model.feature_importances_
        else:
            importance = model.feature_importances_
            
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Display feature importance
        st.sidebar.subheader("Feature Importance")
        st.sidebar.dataframe(feature_importance)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(8, 4))
        feature_importance.plot(kind='bar', x='Feature', y='Importance', ax=ax)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.sidebar.pyplot(fig)

else:
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

    days_ahead = st.slider("Select number of future days", 1, 365, 30)
    forecast_df = pd.DataFrame(columns=['Date', 'Predicted Close'])

    # Store last known values
    last_close_vals = list(df['Close'].values[-10:])
    last_open_vals = list(df['Open'].values[-10:])
    last_volume_vals = list(df['Volume'].values[-10:])
    
    # Function to handle infinite or very large values
    def safe_value(value, default=0.0, max_allowed=1e6):
        if np.isinf(value) or np.isnan(value) or abs(value) > max_allowed:
            return default
        return value

    # Create rolling features for prediction
    for i in range(1, days_ahead + 1):
        try:
            date = df.index[-1] + pd.Timedelta(days=i)
            
            # Predict next open (assuming it's close to last close)
            next_open = last_close_vals[-1]
            
            # Calculate features based on rolling windows
            daily_change = safe_value(last_close_vals[-1] - next_open)
            
            # Avoid division by zero and handle very small numbers
            if abs(next_open) < 1e-6:
                change_pct = 0
            else:
                change_pct = safe_value((daily_change / next_open) * 100)
            
            # Calculate moving averages and std with safety checks
            ma_5 = safe_value(np.mean(last_close_vals[-5:]))
            ma_10 = safe_value(np.mean(last_close_vals[-10:]))
            std_5 = safe_value(np.std(last_close_vals[-5:]))
            
            # Estimate next volume using moving average with bounds
            next_volume = safe_value(
                np.mean(last_volume_vals[-5:]) * (1 + np.random.uniform(-0.1, 0.1)),
                default=np.mean(last_volume_vals[-5:])
            )
            
            # Create feature array for prediction
            features_future = np.array([[
                safe_value(next_open),
                safe_value(next_volume),
                safe_value(daily_change),
                safe_value(change_pct),
                safe_value(ma_5),
                safe_value(ma_10),
                safe_value(std_5)
            ]])
            
            # Ensure no infinite values in features
            features_future = np.clip(features_future, -1e6, 1e6)
            
            # Make prediction
            next_close = forecast_model.predict(features_future)[0]
            
            # Ensure prediction is reasonable
            next_close = safe_value(next_close, default=last_close_vals[-1])
            
            # Update rolling values for next iteration
            last_close_vals.append(next_close)
            last_open_vals.append(next_open)
            last_volume_vals.append(next_volume)
            
            # Store prediction
            row_df = pd.DataFrame([[date, next_close]], columns=['Date', 'Predicted Close'])
            forecast_df = pd.concat([forecast_df, row_df], ignore_index=True)
            
        except Exception as e:
            st.warning(f"Warning: Error in prediction for day {i}. Using last known good value.")
            # Use last known good value
            next_close = last_close_vals[-1]
            row_df = pd.DataFrame([[date, next_close]], columns=['Date', 'Predicted Close'])
            forecast_df = pd.concat([forecast_df, row_df], ignore_index=True)

    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_df.set_index('Date', inplace=True)

    # Add validation for final forecast values
    forecast_df['Predicted Close'] = forecast_df['Predicted Close'].apply(
        lambda x: safe_value(x, default=df['Close'].mean())
    )

    # Combine actual and forecasted data
    combined_df = pd.concat([df[['Close']], forecast_df.rename(columns={'Predicted Close': 'Close'})])

    # Time range filter
    st.subheader("📈 Combined Time Series: Actual + Forecast")

    # Time range filtering
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

    # Prepare combined DataFrame for plotting
    actual_trimmed = df[df.index >= plot_start_date][['Close']].copy()
    forecast_trimmed = forecast_df.copy()

    # Add predicted values
    predicted_values = pd.DataFrame(
        model.predict(df[features]),
        index=df.index,
        columns=['Predicted']
    )
    predicted_trimmed = predicted_values[predicted_values.index >= plot_start_date].copy()

    # Rename columns for the chart
    actual_trimmed.rename(columns={"Close": "Actual"}, inplace=True)
    forecast_trimmed.rename(columns={"Predicted Close": "Forecast"}, inplace=True)

    # Combine all three
    combined_chart_df = pd.concat([
        actual_trimmed, 
        predicted_trimmed,
        forecast_trimmed
    ], axis=1)
    combined_chart_df = combined_chart_df.sort_index()

    # Show in line chart (with legend and Streamlit default color scheme)
    st.line_chart(combined_chart_df)

    # Forecast table
    st.subheader("📋 Forecast Table")
    st.dataframe(forecast_df.style.format({"Predicted Close": "₹{:.2f}"}))

    # Prediction on specific date
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

# Download predictions
st.download_button(
    "📥 Download Model Predictions",
    data=pd.DataFrame({"Actual": y_test.values, "Predicted": predictions}).to_csv(index=False),
    file_name="model_predictions.csv",
    mime="text/csv"
)