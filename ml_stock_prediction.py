import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("stock_analysis_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Select one company for modeling
company = "AAPL"
company_data = df[df["Ticker"] == company].copy()

# Feature Engineering
# Creating lag Features
company_data["Lag_1"] = company_data["Close"].shift(1)
company_data["Lag_2"] = company_data["Close"].shift(2)
company_data["Lag_3"] = company_data["Close"].shift(3)

# Droping rows with NAN due to lag and reset index
company_data = company_data.dropna().reset_index(drop=True)

print("Available columns:", company_data.columns.tolist())

# Defining Features (X) and Target (y)
X = company_data[[
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "7-Days Moving Average",
    "30-Day Moving Average",
    "Volatility (30-day std)"
]]
y = company_data["Close"]

# Train - test split (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)

# 1 Linear regression model

lr_model = LinearRegression()
lr_model.fit(X_train,y_train)
y_pred_lr = lr_model.predict(X_test)

# 2 Random forest Regressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train,y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation Function
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2= r2_score(y_true,y_pred)
    print(f"---  {name} ---")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2 :  {r2:.4f}")
    print("\n")

 # Evaluating model
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

# Plot Actual vs Predicticted
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual", color="black")
plt.plot(y_pred_lr, label="Predicted - Linear Regression", linestyle="--")
plt.plot(y_pred_rf, label="Predicted - Random Forest", linestyle="--")
plt.title(f"{company} - Actual vs Predicted Close Price")
plt.xlabel("Test Data Index")
plt.ylabel("Close Price")
plt.legend()
plt.show()