# Stock Price Prediction- Machine Learning Regression Models

A comprehensive machine learning application for predicting AdaniPorts stock prices using multiple regression models with an interactive Streamlit dashboard.

## Description

This project implements a stock price prediction system for AdaniPorts using various machine learning algorithms. The application features an interactive web interface built with Streamlit that allows users to compare different models, visualize predictions, and forecast future stock prices. The system includes advanced feature engineering with moving averages, volatility indicators, and handles COVID-19 period data smoothing for improved accuracy.

## Features

- **Multiple ML Models**: Compare 8 different regression algorithms
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Support Vector Regressor (SVR)
  - AdaBoost Regressor
  - Gradient Boosting Regressor
  - K-Nearest Neighbors (KNN) Regressor
  - LightGBM Regressor

- **Interactive Dashboard**: Streamlit-based web interface with:
  - Real-time model parameter tuning
  - Time series visualization of closing prices
  - Feature correlation heatmap
  - Model performance comparison
  - Actual vs Predicted price plots

- **Advanced Features**:
  - Feature engineering (Moving Averages, Standard Deviation, Daily Change %)
  - COVID-19 period data smoothing
  - Feature importance visualization for tree-based models
  - Customizable forecast periods (1-365 days)
  - Date-specific price predictions
  - Downloadable forecast and prediction results

- **Model Evaluation Metrics**:
  - R² Score (Coefficient of Determination)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

## Tech Stack

- **Language**: Python 3.x
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn, LightGBM, XGBoost
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Statistical Analysis**: statsmodels, scipy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KanishkaGole/stock-price-prediction-ml-regression.git
cd stock-price-prediction-ml-regression
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have the `ADANIPORTS.csv` dataset in the project root directory

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

4. Use the sidebar to:
   - Select a machine learning model
   - Adjust model parameters (for non-linear models)
   - View feature importance (for tree-based models)

5. Explore the four main tabs:
   - **Dataset**: View feature correlations
   - **Evaluation**: Check model performance metrics
   - **Comparison**: Compare all models side-by-side
   - **Forecast**: Predict future stock prices

## Folder Structure

```
.
├── .vscode/                          # VS Code configuration
├── ML paper/                         # Research papers and references
│   ├── 2310.09903v4.pdf
│   ├── 978-3-031-48781-1_22.pdf
│   ├── AnEmpiricalStudyonImplementationofAIMLinStockMarketPrediction.pdf
│   └── View of STOCK PRICE PREDICTION IN INDIAN MARKETS..._.pdf
├── model_plots/                      # Generated model visualization plots
│   ├── AdaBoost Regressor_*.png
│   ├── Decision Tree_*.png
│   ├── Gradient Boosting_*.png
│   ├── KNN Regressor_*.png
│   ├── Linear Regression_*.png
│   ├── Random Forest_*.png
│   └── Support Vector Regressor_*.png
├── myenv/                            # Virtual environment (ignored)
├── venv/                             # Virtual environment (ignored)
├── ADANIPORTS.csv                    # Stock price dataset
├── app.py                            # Main Streamlit application
├── forecast_v1.py - forecast_v8.py   # Development versions
├── model_comparison.png              # Model comparison visualization
├── requirements.txt                  # Python dependencies
├── Stock_Analysis_Regression_SM.ipynb # Jupyter notebook analysis
├── SM_Final_Sub.ipynb                # Final submission notebook
├── SM_Final report.docx              # Project report
├── ML Paper.doc                      # Research documentation
├── Manthan - Exploratory Data Analysis of Credit Card Fraud Detection.docx
├── .gitignore                        # Git ignore rules
└── README.md                         # Project documentation
```

## Dataset

The project uses historical stock price data for AdaniPorts (`ADANIPORTS.csv`) containing:
- Date
- Open Price
- Close Price
- Volume
- Other OHLC data

## Model Performance

The application provides comprehensive model comparison with metrics including R² Score, MAE, and RMSE. Users can interactively tune hyperparameters and observe real-time performance changes.

## Contributors

- **Kanishka Gole** - [GitHub Profile](https://github.com/KanishkaGole)
- **Jhilik Biswas** 
- **Disha Gaonkar** 

## License

This project is licensed under the MIT License.

---

**Note**: This project is for educational and research purposes. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions.
