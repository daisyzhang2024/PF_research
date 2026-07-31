import pandas as pd

# 1. Read the CSV file
file_path = "combined_LA.csv"  # Replace with your actual CSV file name
df = pd.read_csv(file_path)

# # 2. Drop unnamed / empty trailing columns caused by extra commas
# df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# 3. Convert 'date_dt' column to pandas datetime
df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")

# (Optional) Format the date as 'YYYY-MM-DD' if you don't need the 00:00 timestamp
# df['date_dt'] = df['date_dt'].dt.strftime('%Y-%m-%d')
# Drop rows where every single column is NaN / empty
df = df.dropna(how="all")
# 4. View the cleaned data info to verify datatypes
print("--- Data Info ---")
print(df.info())
print("\n--- Preview ---")
print(df.head())

# 5. Save back to a cleaned CSV without trailing comma artifacts
df.to_csv("combined_LA.csv", index=False)
print("\nCleaned CSV saved to 'combined_LA' successfully!")