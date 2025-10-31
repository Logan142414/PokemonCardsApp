import streamlit as st         # for the web app
import pandas as pd             # for DataFrame manipulation
import os                       # to access environment variables (HF_TOKEN)
from typing import Optional, List  # for your HFInferenceLLM type hints
from pydantic import Field      # for your HFInferenceLLM class
from langchain.llms.base import LLM  # base class for your HFInferenceLLM
from huggingface_hub import InferenceClient  # to call Hugging Face API
import time                     # for sleep between requests
import requests                 # for HTTP requests
from bs4 import BeautifulSoup   # for HTML parsing
from datetime import datetime   # for timestamps
from zoneinfo import ZoneInfo   # for timezone-aware timestamps
from google.cloud import storage
import json
from io import BytesIO
from langchain.agents import create_pandas_dataframe_agent
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import plotly.express as px



# --------------------------
# Google Cloud Storage Setup
# --------------------------
from google.oauth2 import service_account

# Load service account JSON from Streamlit secrets
gcp_credentials = service_account.Credentials.from_service_account_info(
    json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
)

# Initialize client
storage_client = storage.Client(credentials=gcp_credentials)

# Set your bucket name
BUCKET_NAME = "pokemon-price-history-data" 


# --------------------------
# Web page set up, colors, details, etc
# --------------------------

st.markdown(
    """
    <style>
    /* Global layout */
    body, .stApp {
        background-color: #2a2a2a;  /* Brighter dark gray */
        color: #FFFFFF;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Main headers */
    h1, h2, h3, h4, h5 {
        color: #FFFFFF;
        margin: 0;
        padding-bottom: 0.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #800000;  /* Maroon sidebar */
        color: white;
    }

    /* Sidebar slider container */
    section[data-testid="stSidebar"] .stSlider {
        background-color: #800000 !important;  /* Match sidebar */
        padding: 0.5rem;
        border-radius: 6px;
    }

    /* Slider track and handle */
    .stSlider .rc-slider-track {
        background-color: #ffcccc !important;
    }

    .stSlider .rc-slider-handle {
        background-color: white !important;
        border: 2px solid #ff6666 !important;
    }

    .stSlider .rc-slider-mark-text {
        color: white !important;
        font-weight: bold;
    }

    /* Sidebar label text */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .st-c2,
    section[data-testid="stSidebar"] .st-c3,
    section[data-testid="stSidebar"] .stCheckbox > label {
        color: white !important;
        font-weight: 500;
    }

    /* Inputs and checkboxes in sidebar */
    section[data-testid="stSidebar"] .stCheckbox,
    section[data-testid="stSidebar"] .stTextInput,
    section[data-testid="stSidebar"] .stNumberInput {
        color: white !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #444 !important;
        color: white !important;
        border: 1px solid #666;
        border-radius: 6px;
    }

    .stButton > button:hover {
        background-color: #666 !important;
        color: #ffffff !important;
    }

    /* Download Button */
    .stDownloadButton > button {
        background-color: #444 !important;
        color: white !important;
        border-radius: 6px;
    }

    /* DataFrame styling */
    .stDataFrame {
        background-color: #1e1e1e !important;
        color: white !important;
        border-radius: 6px;
        border: 1px solid #333;
        padding: 10px;
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #1f77b4 !important;
    }

    /* Caption / Footnotes */
    .stMarkdown small, .stCaption {
        color: #CCCCCC;
    }

    /* Make the label text for text_input white */
    label[for="Ask a question about the dataset:"] {
        color: white !important;
    }

   section[data-testid="stSidebar"] input[type="text"],
    section[data-testid="stSidebar"] input[type="number"],
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stNumberInput input {
        color: black !important;
        background-color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
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
            sorted_url = f"{url}?sort=highest-price"
            res = requests.get(sorted_url, headers=headers)
            soup = BeautifulSoup(res.text, 'html.parser')
            rows = soup.select('table tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 5:
                    img_tag = cols[0].find("img")

                    if img_tag and "src" in img_tag.attrs:
                        img_url = img_tag["src"]
                    else:
                        img_url = ""

                    name = cols[1].text.strip()
                    ungraded = cols[2].text.strip().replace("$", "").replace(",", "")
                    grade9 = cols[3].text.strip().replace("$", "").replace(",", "")
                    psa10 = cols[4].text.strip().replace("$", "").replace(",", "")

                    all_data.append({
                        "Set": url.split('/')[-1],
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

    # Turn into DataFrame
    df = pd.DataFrame(all_data)

    if df.empty:
        return df

    # Ensure correct column types
    for col in ["Ungraded_Price", "Grade_9_Price", "PSA_10_Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Deal_Value"] = df["Grade_9_Price"] - df["Ungraded_Price"]
    df["Set"] = df["Set"].str.replace("pokemon-", "", regex=False)



    return df



# --------------------------
@st.cache_data
def load_data():
    file_path = "data/latest_pokemon_prices.csv"
    expected_cols = {"Set", "Card_Name", "Ungraded_Price", "Grade_9_Price", "PSA_10_Price", "Image_URL"}

    if not os.path.exists(file_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)

        if not expected_cols.issubset(df.columns):
            os.remove(file_path)
            return pd.DataFrame()

        return df

    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()


# --------------------------
# App UI
# --------------------------
# Title with logos on both sides
# App UI: Title with logos
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.image("pikachu-running.png", use_container_width=True)

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
    st.image("Pokeball-removebg-preview.png", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------
# Show refresh button
if st.button("Refresh Price Data"):
    with st.spinner("Scraping all Pokémon card sets (this may take up to 5 minutes)..."):
        df_scraped = scrape_pricecharting_data()

        if not df_scraped.empty:
            # Add today's date column
            today = datetime.now().strftime("%Y-%m-%d")
            df_scraped["Date"] = today

            # --------------------------
            # Save/update growing history file on Google Cloud Storage
            # --------------------------
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob("pokemon_price_history.csv")

            # Try to load old data from GCS
            try:
                old_data = blob.download_as_bytes()
                old = pd.read_csv(BytesIO(old_data))
            except Exception:
                old = pd.DataFrame()

            # Combine if new date not already included
            if old.empty:
                combined = df_scraped
            elif today not in old["Date"].astype(str).values:
                combined = pd.concat([old, df_scraped], ignore_index=True)
            else:
                combined = old  # keep full history if today's data already exists

            # Upload updated file back to GCS
            csv_buffer = BytesIO()
            combined.to_csv(csv_buffer, index=False)
            blob.upload_from_file(csv_buffer, content_type="text/csv", rewind=True)

            st.success("✅ Data refreshed and uploaded to cloud!")
            df = df_scraped  # keep this line

        else:
            st.error("Scraping failed or returned no data.")
            st.stop()

# --------------------------
# Load existing data if not refreshed
@st.cache_data
def get_valid_data(today: str):
    df = load_data()  # load latest_pokemon_prices.csv

    required_cols = {"Ungraded_Price", "Grade_9_Price", "PSA_10_Price"}
    if df.empty or not required_cols.issubset(df.columns):
        df = scrape_pricecharting_data()
        if not df.empty:
            os.makedirs("data", exist_ok=True)
            df.to_csv("data/latest_pokemon_prices.csv", index=False)
        else:
            st.error("Scraping failed. Please try again.")
            st.stop()

    return df

today = datetime.now().strftime("%Y-%m-%d")
df = get_valid_data(today)

# --------------------------
# Process columns
df["Deal_Value"] = df["Grade_9_Price"] - df["Ungraded_Price"]
df["Set"] = df["Set"].str.replace("pokemon-", "", regex=False)

eastern_time = datetime.now(ZoneInfo("America/New_York"))
st.caption(f"🕒 Data last updated: {eastern_time.strftime('%Y-%m-%d %H:%M')}")

# Convert price columns to numeric if needed
price_cols = ["Ungraded_Price", "Grade_9_Price", "PSA_10_Price"]
for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Optional visual preview of each card
#show_visuals = st.checkbox("Show visual preview of each card", value=False)

#if show_visuals:
  #  st.markdown("### 📸 Visual Results")
   # for _, row in filtered.iterrows():
      #  if pd.notna(row["Image_URL"]) and row["Image_URL"].strip() != "":
           # st.image(row["Image_URL"], width=100)
       # st.write(f"**{row['Card_Name']}**")
       # st.write(
           # f"Ungraded: ${row['Ungraded_Price']:.2f} | "
           # f"PSA 9: ${row['Grade_9_Price']:.2f} | "
           # f"PSA 10: ${row['PSA_10_Price']:.2f} | "
           # f"Deal Value: ${row['Deal_Value']:.2f}"
       # )
       # st.markdown("---")

# --------------------------
# Download Button
# --------------------------
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob("pokemon_price_history.csv")
try:
    data = blob.download_as_bytes()
    history_df = pd.read_csv(BytesIO(data))
    st.success("")
except Exception as e:
    st.warning(f"No existing cloud history found or failed to load: {e}")
    history_df = pd.DataFrame()

# Work with full history + keep latest snapshot separate
if not history_df.empty:
    history_df["Date"] = pd.to_datetime(history_df["Date"])
    latest_date = history_df["Date"].max()
    latest_df = history_df[history_df["Date"] == latest_date].copy()
else:
    latest_df = pd.DataFrame()

df = latest_df.copy()

# Compute 3, 7, 14, 30 day price changes inside the full history 
if not history_df.empty:
    latest_date = history_df["Date"].max()

    for days in [3, 7, 14, 30]:
        target_date = latest_date - pd.Timedelta(days=days)

        # Define ±1 day window around the target date
        lower_bound = target_date - pd.Timedelta(days=1)
        upper_bound = target_date + pd.Timedelta(days=1)

        # Keep only records in that date range
        window_df = history_df[
            (history_df["Date"] >= lower_bound) &
            (history_df["Date"] <= upper_bound)
        ].copy()

        if not window_df.empty:
            # Pick record closest to the target date per card
            window_df["date_diff"] = (window_df["Date"] - target_date).abs()
            closest = (
                window_df.sort_values("date_diff")
                .groupby(["Set", "Card_Name"])
                .first()
                .reset_index()
            )

            # Merge the closest price back into main history
            history_df = pd.merge(
                history_df,
                closest[["Set", "Card_Name", "Ungraded_Price"]],
                on=["Set", "Card_Name"],
                how="left",
                suffixes=("", f"_{days}d_ago")
            )

            # Compute change vs. current price
            history_df[f"Ungraded_{days}d_Change"] = (
                history_df["Ungraded_Price"] - history_df[f"Ungraded_Price_{days}d_ago"]
            )

    # Keep only the latest snapshot for export / summary
    latest_with_changes = history_df[history_df["Date"] == latest_date].copy()
    latest_with_changes = latest_with_changes.loc[:, ~latest_with_changes.columns.duplicated()]
else:
    latest_with_changes = pd.DataFrame()
    
st.session_state["latest_with_changes"] = latest_with_changes

# --------------------------
# Filter Controls
# --------------------------
st.sidebar.header("Filter Options")

all_sets = sorted(latest_with_changes["Set"].unique())
select_all_sets = st.sidebar.checkbox("Select all sets", value=True)

if select_all_sets:
    selected_sets = all_sets
else:
    selected_sets = st.sidebar.multiselect("Choose Set(s)", options=all_sets)

if not selected_sets:
    st.warning("Please select at least one set.")
    st.stop()

min_ungraded = st.sidebar.number_input("Min price ($)", min_value=0.01, max_value=10000.0, value=0.01, step=0.01)
max_ungraded = st.sidebar.number_input("Max price ($)", min_value=0.01, max_value=10000.0, value=10000.0, step=0.01)
min_grade9 = st.sidebar.number_input("Min Grade 9 Price", min_value=0, value=0)
min_psa10 = st.sidebar.number_input("Min PSA 10 Price", min_value=0, value=0)

# --------------------------
# 3-day, 7-day, 14-day, and 30-day Ungraded Price Change Filters
change_filters = {}
for days in [3, 7, 14, 30]:
    col_name = f"Ungraded_{days}d_Change"
    min_val, max_val = st.sidebar.slider(
        f"{days}-Day Ungraded Price Change ($)",
        min_value=-1000.0,
        max_value=1000.0,
        value=(-1000.0, 1000.0),
        step=0.01
    )
    change_filters[col_name] = (min_val, max_val)

# --------------------------
# Apply all filters
filtered = latest_with_changes[
    (latest_with_changes["Set"].isin(selected_sets)) &
    (latest_with_changes["Ungraded_Price"].between(min_ungraded, max_ungraded)) &
    (latest_with_changes["Grade_9_Price"] >= min_grade9) &
    (latest_with_changes["PSA_10_Price"] >= min_psa10)
]

for col_name, (min_val, max_val) in change_filters.items():
    if col_name in filtered.columns:
        # Determine if user has changed slider from default
        include_na = (min_val <= -1000 and max_val >= 1000)
        
        if include_na:
            filtered = filtered[
                filtered[col_name].isna() | filtered[col_name].between(min_val, max_val)
            ]
        else:
            filtered = filtered[
                filtered[col_name].between(min_val, max_val)
            ]

st.subheader(f"Filtered Results ({len(filtered)} cards)")

st.sidebar.caption("⚠️ Enabling image thumbnails may slow down loading or cause images to fail.")
show_images = st.sidebar.checkbox("Show image thumbnails", value=False)

if show_images:
    def image_tag(url):
        return f'<a href="{url}" target="_blank"><img src="{url}" width="80"></a>'

    styled_df = filtered.copy()
    styled_df["Image"] = styled_df["Image_URL"].apply(image_tag)
    styled_df = styled_df.drop(columns=["Image_URL"])  

    cols = ["Image"] + [col for col in styled_df.columns if col != "Image"]
    styled_df = styled_df[cols]

    st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    filtered_display = filtered.drop(columns=["Image_URL"], errors="ignore")
    st.dataframe(filtered_display.reset_index(drop=True))





# --------------------------
# Download Dataset as CSV
# --------------------------

# Apply filters to the full history (not just latest snapshot)
history_filtered = history_df[
    (history_df["Set"].isin(selected_sets)) &
    (history_df["Ungraded_Price"].between(min_ungraded, max_ungraded)) &
    (history_df["Grade_9_Price"] >= min_grade9) &
    (history_df["PSA_10_Price"] >= min_psa10)
]

csv_data = convert_df_to_csv(history_filtered)

now = datetime.now().strftime("%Y-%m-%d_%H-%M")
file_name = f"history_filtered_cards_{now}_UG{min_ungraded}-{max_ungraded}_G9{min_grade9}_P10{min_psa10}.csv"

st.download_button(
    label="Download all-time data as CSV (using current filters)",
    data=csv_data,
    file_name=file_name,
    mime="text/csv"
)



# --------------------------
# 📊 Price Trend Analysis
# --------------------------

st.markdown("---")
st.subheader("Visualizations")

# Only include periods where the column exists
avg_changes = {}
for days, col_name in [
    ('3-day', 'Ungraded_3d_Change'),
    ('7-day', 'Ungraded_7d_Change'),
    ('14-day', 'Ungraded_14d_Change'),
    ('30-day', 'Ungraded_30d_Change')
]:
    if col_name in filtered.columns:
        # Use .dropna() to exclude NaN values before calculating mean
        valid_values = filtered[col_name].dropna()
        if len(valid_values) > 0:  # Only add if there's at least one non-NaN value
            avg_changes[days] = round(valid_values.mean(), 2)  # Round to 2 decimals

# Only create chart if we have any data
if avg_changes:
    fig = px.bar(
        x=list(avg_changes.keys()),
        y=list(avg_changes.values()),
        labels={'x': 'Time Period', 'y': 'Average Price Change ($)'},
        title='Average Price Changes by Period'
    )
    
    # Add subtitle explaining that filters affect this chart
    st.caption("💡 Both charts reflect the filters applied above (set, price range, etc.)")
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("📊 Price change data will appear once you have multiple days of historical data.")

# second graph looking at average ungraded card prices

if "Date" in history_df.columns and "Ungraded_Price" in history_df.columns:
    # Apply the same filters to historical data
    history_filtered = history_df[
        (history_df["Set"].isin(selected_sets)) &
        (history_df["Ungraded_Price"].between(min_ungraded, max_ungraded)) &
        (history_df["Grade_9_Price"] >= min_grade9) &
        (history_df["PSA_10_Price"] >= min_psa10)
    ]

    # Group by date and compute average ungraded price
    trend_data = (
        history_filtered.groupby("Date")["Ungraded_Price"]
        .mean()
        .reset_index()
    )

    if not trend_data.empty:
        fig = px.line(
            trend_data,
            x="Date",
            y="Ungraded_Price",
            title="Average Ungraded Price Over Time",
            labels={"Date": "Date", "Ungraded_Price": "Average Price ($)"}
        )
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical price data available for the current filters.")
else:
    st.info("Historical price data unavailable or missing required columns.")




