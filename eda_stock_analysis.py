import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading dataset
df= pd.read_csv("stock_analysis_data.csv")

# Stock Price Trends
plt.figure(figsize=(12,6))
for company in df["Ticker"].unique():
    company_data = df[df["Ticker"] == company]
    plt.plot(company_data["Date"], company_data["Close"], label=company)

plt.title("Stock Price Trends (Last 3 Months)")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.show()

# Moving Averages

plt.figure(figsize=(12,6))
company = "AAPL"
company_data = df[df["Ticker"] == company]

plt.plot(company_data["Date"], company_data["Close"], label="Close Price", alpha=0.7)
plt.plot(company_data["Date"], company_data["7-Days Moving Average"], label="7_Day MA")
plt.plot(company_data["Date"], company_data["30-Day Moving Average"], label="30-Day MA")

plt.title(f"{company} - Moving Average")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


# Volatility (30- Days rolling std of returns)

plt.figure(figsize=(12,6))
for company in df["Ticker"].unique():
    company_data = df[df["Ticker"] == company]
    plt.plot(company_data["Date"], company_data["Volatility (30-day std)"], label=company)

plt.title("Volatility Comparison")
plt.xlabel("Date")
plt.ylabel("Volatility (Std of Returns)")
plt.legend()
plt.show()

# Distribution of Daily Returns

plt.figure(figsize=(12,6))
for company in df["Ticker"].unique():
    sns.histplot(df[df["Ticker"] == company]["Daily Return (%)"], bins=50, kde=True, label=company, alpha=0.5)

plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Correlation Heatmap
#Pivoting to wide format (Date X Company Close Prices)
pivot_df = df.pivot(index="Date", columns="Ticker", values="Close")
corr_matrix = pivot_df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Between Stock Prices")
plt.show()
