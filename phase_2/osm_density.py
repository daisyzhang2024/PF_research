import requests
import time
import pandas as pd

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

HEADERS = {
    "User-Agent": "PF_research CHMR project (student research, contact: daisyz@uchicago.edu)"
}


def query_shop_count(date_str, shop_filter=""):
    """
    date_str: e.g. '2019-04-01'
    shop_filter: optional regex filter for shop tag values, e.g. '~"^(supermarket|convenience)$"'
    """
    q = f"""
    [out:json][timeout:180][date:"{date_str}T00:00:00Z"];
    area["name"="Cook County"]["admin_level"="6"]->.a;
    (
      node["shop"{shop_filter}](area.a);
      way["shop"{shop_filter}](area.a);
    );
    out count;
    """

    r = requests.post(OVERPASS_URL, data={"data": q}, headers=HEADERS)

    if r.status_code != 200:
        print(f"Error {r.status_code} for {date_str}: {r.text[:300]}")

    r.raise_for_status()

    return int(r.json()["elements"][0]["tags"]["total"])


def main():
    months = pd.date_range("2015-01-01", "2021-12-01", freq="MS")
    records = []

    for m_date in months:
        date_str = m_date.strftime("%Y-%m-%d")
        try:
            count = query_shop_count(date_str)
            print(f"{date_str}: {count} shops")
            records.append({"date": m_date, "osm_shop_count": count})
        except Exception as e:
            print(f"Failed on {date_str}: {e}")
            records.append({"date": m_date, "osm_shop_count": None})

        time.sleep(10)  # be polite to the public Overpass instance

    osm_monthly = pd.DataFrame(records)
    osm_monthly["osm_net_new"] = osm_monthly["osm_shop_count"].diff()
    osm_monthly.to_csv("osm_monthly.csv", index=False)
    print("Saved osm_monthly.csv")


if __name__ == "__main__":
    main()