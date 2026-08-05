import pandas as pd

# Load the CSV file
df = pd.read_csv("MTA.csv")

# Save it as an Excel file
df.to_excel("MTA.xlsx", index=False)