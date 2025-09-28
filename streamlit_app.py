import streamlit as st
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os
from zoneinfo import ZoneInfo

# --------------------------
# Scraping Logic
# --------------------------
def scrape_pricecharting_data():
    BASE_URL = "https://www.pricecharting.com"
    CATEGORY_URL = f"{BASE_URL}/category/pokemon-cards"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        res = requests.get(CATEGORY_URL, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
    except Exception as e:
        st.error("Error fetching category page.")
        return pd.DataFrame()

    # Get all set links
    set_links = soup.select('a[href^="/console/pokemon"]')
    set_urls = list(set(BASE_URL + link["href"] for link in set_links))
    set_urls = [url for url in set_urls if "japanese" not in url.lower()]  # skip Japanese sets

    all_data = []
    progress = st.progress(0)
    for i, url in enumerate(set_urls):
        try:
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, 'html.parser')

            rows = soup.select('table tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 5:
                    img_tag = cols[0].find("img")
                    img_url = img_tag["src"] if img_tag and "src" in img_tag.attrs else ""

                    name = cols[1].text.strip()
                    ungraded = cols[2].text.strip().replace("$", "").replace(",", "")
                    grade9 = cols[3].text.strip().replace("$", "").replace(",", "")
                    psa10 = cols[4].text.strip().replace("$", "").replace(",", "")

                    all_data.append({
                        "Set": url.split('/')[-1].replace("pokemon-", ""),
                        "Card_Name": name,
                        "Ungraded_Price": ungraded,
                        "Grade_9_Price": grade9,
                        "PSA_10_Price": psa10,
                        "Image_URL": img_url
                    })
        except Exception as e:
            st.warning(f"Error scraping {url}: {e}")
            continue

        progress.progress((i + 1) / len(set_urls))
        time.sleep(0.3)

    df = pd.DataFrame(all_data)
    if df.empty:
        return df

    # Ensure correct column types
    for col in ["Ungraded_Price", "Grade_9_Price", "PSA_10_Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Deal_Value"] = df["Grade_9_Price"] - df["Ungraded_Price"]
    return df


# --------------------------
# Data Management
# --------------------------
@st.cache_data
def load_latest():
    """Load the latest snapshot for fast startup"""
    file_path = "data/latest_pokemon_prices.csv"
    if not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except:
        return pd.DataFrame()

def append_to_history(df_scraped):
    """Append today's scrape to the history file if not already present"""
    today = datetime.now().strftime("%Y-%m-%d")
    df_scraped["Date"] = today

    os.makedirs("data", exist_ok=True)
    history_path = "data/pokemon_price_history.csv"

    if os.path.exists(history_path):
        old = pd.read_csv(history_path)
        if today in old["Date"].astype(str).values:
            # ✅ Safeguard: skip duplicate for today
            return old
        combined = pd.concat([old, df_scraped], ignore_index=True)
    else:
        combined = df_scraped

    combined.to_csv(history_path, index=False)
    return combined


def load_with_history():
    """Load latest data, and add 7-day price change if history available"""
    df = load_latest()
    history_path = "data/pokemon_price_history.csv"

    if os.path.exists(history_path):
        history = pd.read_csv(history_path)
        history["Date"] = pd.to_datetime(history["Date"])

        latest_date = history["Date"].max()
        latest = history[history["Date"] == latest_date]

        # Try to get 7-day old snapshot
        target_date = latest_date - pd.Timedelta(days=7)
        past = history[history["Date"] == target_date]

        if not past.empty:
            df = latest.merge(
                past,
                on=["Set", "Card_Name"],
                suffixes=("", "_7dAgo")
            )
            df["7d_Ungraded_Change"] = df["Ungraded_Price"] - df["Ungraded_Price_7dAgo"]

    return df


# --------------------------
# UI – Refresh Button
# --------------------------
if st.button("Refresh Price Data"):
    with st.spinner("Scraping Pokémon card sets..."):
        df_scraped = scrape_pricecharting_data()
        if not df_scraped.empty:
            # Save to history (with safeguard)
            history = append_to_history(df_scraped)
            # Always update latest snapshot
            df_scraped.to_csv("data/latest_pokemon_prices.csv", index=False)
            st.success("Data refreshed!")
            df = df_scraped
        else:
            st.error("Scraping failed.")
            st.stop()

# --------------------------
# Load data
# --------------------------
df = load_with_history()
if df.empty:
    st.error("No data available. Please refresh first.")
    st.stop()

# Show last updated
eastern_time = datetime.now(ZoneInfo("America/New_York"))
st.caption(f"🕒 Data last updated: {eastern_time.strftime('%Y-%m-%d %H:%M')}")

# --------------------------
# Sidebar filters (including new 7-day filter)
# --------------------------
st.sidebar.header("Filter Options")

# Price filters...
min_change = st.sidebar.number_input("Min 7-day Price Change ($)", value=-100)
max_change = st.sidebar.number_input("Max 7-day Price Change ($)", value=100)

if "7d_Ungraded_Change" in df.columns:
    df = df[df["7d_Ungraded_Change"].between(min_change, max_change)]
