import pandas as pd

# 1. Load both CSV files
combined_df = pd.read_csv("combined_CC_v3.csv")
ohsome_df = pd.read_csv("osm_monthly_ohsome_2015_2021.csv")
sent_df = pd.read_csv("chicago_sentiment.csv")

# 2. Convert date columns to uniform datetime objects for a clean join
combined_df["date_dt"] = pd.to_datetime(combined_df["date"])
ohsome_df["date_dt"] = pd.to_datetime(ohsome_df["date"])
sent_df["date_dt"] = pd.to_datetime(sent_df["date"])

# 3. Left join Ohsome, sentiment data onto the existing combined dataset
merged_df = pd.merge(
    combined_df,
    ohsome_df[["date_dt", "osm_shop_count", "osm_net_new"]],
    on="date_dt",
    how="left"
)
merged_df = pd.merge(
    merged_df,
    sent_df[["date_dt", "chicago_search_sentiment"]],
    on="date_dt",
    how="left"
)

# 4. Clean up the temporary datetime key column
merged_df.drop(columns=["date"], inplace=True)

# Make date column first column
first_col = merged_df.pop('date_dt')
merged_df.insert(0, 'date_dt', first_col)

# 5. Export updated dataset
output_file = "combined_CC_v4.csv"
merged_df.to_csv(output_file, index=False)
merged_df.to_excel("combined_CC_v4.xlsx", index=False)

print(f"Successfully merged data! Output saved to '{output_file}'\n")
print(merged_df.head(10))