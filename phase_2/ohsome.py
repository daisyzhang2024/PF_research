import pandas as pd
import requests

OHSOME_URL = "https://api.ohsome.org/v1/elements/count"

# Chicago bounding box: min_lon, min_lat, max_lon, max_lat
CHICAGO_BBOX = "-87.940248,41.644286,-87.524044,42.023135"

params = {
    "bboxes": CHICAGO_BBOX,
    "filter": "shop=* and (type:node or type:way or type:relation)",
    "time": "2015-01-01/2026-04-01/P1M",  # From Jan 2015 to Apr 2026 (monthly)
    "format": "json",
}

print("Querying Ohsome API for 2015–2026 shop counts...")

try:
    response = requests.post(OHSOME_URL, data=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    records = []
    for entry in data.get("result", []):
        records.append({
            "date": entry["timestamp"],
            "osm_shop_count": entry["value"]
        })

    df = pd.DataFrame(records)
    
    # Clean up dates and compute net changes
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["osm_net_new"] = df["osm_shop_count"].diff()

    # Save output
    output_filename = "osm_monthly_ohsome_2015_2021.csv"
    df.to_csv(output_filename, index=False)
    
    print(f"\nDone! Successfully pulled {len(df)} months of data.")
    print(f"Saved to {output_filename}\n")
    print(df.head(12))  # Display the first year of data

except requests.exceptions.ConnectionError as e:
    print(f"\nNetwork error: Could not reach Ohsome API. Check your VPN/firewall.\n{e}")
except requests.exceptions.HTTPError as e:
    print(f"\nAPI Error: {response.status_code} - {response.text}")