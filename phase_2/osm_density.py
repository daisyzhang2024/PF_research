import osmnx as ox
import pandas as pd
import time

PLACE = "Chicago, Illinois, USA"
TAGS = {"shop": True}

MAX_RETRIES = 5
BASE_BACKOFF = 20  # seconds, doubles each retry

# Rotate through these if one backend refuses connections or errors out.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]

ox.settings.requests_timeout = 180
ox.settings.log_console = False
ox.settings.use_cache = True  # cache successful responses locally so re-runs don't re-hit the server


def set_historical_date(date_str):
    """date_str: e.g. '2015-01-01'"""
    ox.settings.overpass_settings = (
        f'[out:json][timeout:{ox.settings.requests_timeout}]'
        f'[date:"{date_str}T00:00:00Z"]'
    )


def query_shop_count_for_date(date_str):
    set_historical_date(date_str)

    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        # Cycle through endpoints across attempts so a dead backend gets skipped.
        endpoint = OVERPASS_ENDPOINTS[(attempt - 1) % len(OVERPASS_ENDPOINTS)]
        ox.settings.overpass_url = endpoint

        start = time.monotonic()
        try:
            gdf = ox.features_from_place(PLACE, TAGS)
            elapsed = time.monotonic() - start
            count = len(gdf)
            print(f"  [{date_str}] {elapsed:.1f}s via {endpoint} -> {count} shop features")
            return count
        except Exception as e:
            elapsed = time.monotonic() - start
            last_exception = e
            wait = BASE_BACKOFF * (2 ** (attempt - 1))
            print(f"  [{date_str}] error after {elapsed:.1f}s via {endpoint}: {e} "
                  f"-- retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts for {date_str}: {last_exception}")


def main():
    months = pd.date_range("2015-01-01", "2021-12-01", freq="MS")
    records = []

    run_start = time.monotonic()

    for m_date in months:
        date_str = m_date.strftime("%Y-%m-%d")
        try:
            count = query_shop_count_for_date(date_str)
            records.append({"date": m_date, "osm_shop_count": count})
        except Exception as e:
            print(f"Failed on {date_str}: {e}")
            records.append({"date": m_date, "osm_shop_count": None})

        # Save incrementally so a crash partway through doesn't lose progress.
        pd.DataFrame(records).to_csv("osm_monthly_osmnx.csv", index=False)

        time.sleep(5)

    total_elapsed = time.monotonic() - run_start
    osm_monthly = pd.DataFrame(records)
    osm_monthly["osm_net_new"] = osm_monthly["osm_shop_count"].diff()
    osm_monthly.to_csv("osm_monthly_osmnx.csv", index=False)
    print(f"Saved osm_monthly_osmnx.csv ({len(records)} months in {total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()