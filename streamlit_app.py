import streamlit as st
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# --------------------------
#Scraping Logic


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

    # Remove Japanese sets for now. Scrape takes too long otherwise
    set_urls = [url for url in set_urls if "japanese" not in url.lower()]

    all_data = []
    
    progress = st.progress(0)
    for i, url in enumerate(set_urls):
        try:
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, 'html.parser')

            rows = soup.select('table tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    name = cols[1].text.strip()
                    ungraded = cols[2].text.strip().replace("$", "").replace(",", "")
                    grade9 = cols[3].text.strip().replace("$", "").replace(",", "")
                    psa10 = cols[4].text.strip().replace("$", "").replace(",", "")

                    all_data.append({
                        "Set": url.split('/')[-1],
                        "Card_Name": name,
                        "Ungraded_Price": ungraded,
                        "Grade_9_Price": grade9,
                        "PSA_10_Price": psa10
                    })

        except Exception as e:
            st.warning(f"Error scraping {url}: {e}")
            continue

        progress.progress((i + 1) / len(set_urls))
        time.sleep(0.3)

    # Turn into DataFrame
    df = pd.DataFrame(all_data)

    if df.empty:
        return df

    # Ensure correct column types
    for col in ["Ungraded_Price", "Grade_9_Price", "PSA_10_Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Deal_Value"] = df["Grade_9_Price"] - df["Ungraded_Price"]

    return df



# --------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("latest_pokemon_prices.csv")
        
        # Check for correct columns
        expected_cols = {"Set", "Card_Name", "Ungraded_Price", "Grade_9_Price", "PSA_10_Price"}
        if not expected_cols.issubset(df.columns):
            st.warning("CSV loaded but has missing or unexpected columns.")
            return pd.DataFrame()
        
        if df.empty:
            st.warning("CSV file loaded but it's empty.")
            return pd.DataFrame()

        return df

    except FileNotFoundError:
        st.info("No cached CSV found yet. Please click 'Refresh Price Data'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()


# --------------------------
# App UI
# --------------------------
# Title with logos on both sides
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.image("PikaRunning.png", use_container_width = True)

with col2:
    st.markdown(
        "<h1 style='text-align: center;'>Pokémon Card Price Explorer</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
    "<div style='text-align: center; font-size: 14px;'>Search Pokémon card prices scraped from PriceCharting and find undervalued cards.</div>",
    unsafe_allow_html=True
)

with col3:
    st.image("Pokeball-removebg-preview.png", use_container_width = True)

st.markdown("<br>", unsafe_allow_html=True)

# Load or scrape data
df = load_data()

if st.button("Refresh Price Data"):
    with st.spinner("Scraping all Pokémon card sets (this may take up to 5 minutes)..."):
        df = scrape_pricecharting_data()
        if not df.empty:
            df.columns = df.columns.str.replace(" ", "_") 
            df.to_csv("latest_pokemon_prices.csv", index=False)
            st.success("Data refreshed!")
        st.rerun()  # 👈 force a full rerun of the app

else:
    df = load_data()

    st.caption(f"🕒 Data last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

if df.empty:
    st.warning("No data available. Try refreshing or upload a CSV.")
    st.stop()

# Convert price columns to numeric if needed
price_cols = ["Ungraded_Price", "Grade_9_Price", "PSA_10_Price"]
for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --------------------------
# Filter Controls
# --------------------------
st.sidebar.header("Filter Options")

all_sets = df["Set"].sort_values().unique()
selected_sets = st.sidebar.multiselect("Select Set(s)", all_sets, default=all_sets)

min_ungraded, max_ungraded = st.sidebar.slider("Ungraded Price ($)", min_value=0.01, max_value=1000.0, value=(0.01, 50.0), step=0.01)
min_grade9 = st.sidebar.number_input("Min Grade 9 Price", min_value=0, value=0)
min_psa10 = st.sidebar.number_input("Min PSA 10 Price", min_value=0, value=0)

filtered = df[
    (df["Set"].isin(selected_sets)) &
    (df["Ungraded_Price"].between(min_ungraded, max_ungraded)) &
    (df["Grade_9_Price"] >= min_grade9) &
    (df["PSA_10_Price"] >= min_psa10)
]

st.subheader(f"Filtered Results ({len(filtered)} cards)")
st.dataframe(filtered.reset_index(drop=True))

# --------------------------
# Download Button
# --------------------------
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

csv_data = convert_df_to_csv(filtered)

now = datetime.now().strftime("%Y-%m-%d_%H-%M")
file_name = f"filtered_cards_{now}_UG{min_ungraded}-{max_ungraded}_G9{min_grade9}_P10{min_psa10}.csv"

st.download_button(
    label="Download filtered data as CSV",
    data=csv_data,
    file_name=file_name,
    mime="text/csv"
)

st.markdown("---")
st.markdown("Built by Logan Laszewski")
