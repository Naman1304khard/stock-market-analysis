import pandas as pd

# Loading existing dataset
df = pd.read_csv( "stocks.csv") 

df["Date"] = pd.to_datetime(df["Date"])

#Sorting data by company and date
df = df.sort_values(by=["Ticker","Date"])

print(df.head())

# Calculate derived features for each company seperately
df["Daily Return (%)"] = df.groupby("Ticker")["Close"].pct_change()* 100 
df["7-Days Moving Average"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=7).mean())
df["30-Day Moving Average"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=30).mean())
df["Volatility (30-day std)"] = df.groupby("Ticker")["Daily Return (%)"].transform(lambda x: x.rolling(window=30).std())

# Saving the updated dataset
df.to_csv("stock_analysis_data.csv", index=False)

print("Updated dataset with derived columns saved as 'stock_analysis_data.csv'")