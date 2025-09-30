import streamlit as st
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain.agents import create_pandas_dataframe_agent

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

    /* Force input text color black */
    input[type="number"], input[type="text"] {
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
            res = requests.get(url, headers=headers)
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
            os.makedirs("data", exist_ok=True)

            # Add today's date column
            today = datetime.now().strftime("%Y-%m-%d")
            df_scraped["Date"] = today

            # Save/update growing history file
            history_path = "data/pokemon_price_history.csv"
            if os.path.exists(history_path):
                old = pd.read_csv(history_path)
                # Only append if this date isn’t already in history
                if today not in old["Date"].astype(str).values:
                    combined = pd.concat([old, df_scraped], ignore_index=True)
                    combined.to_csv(history_path, index=False)
            else:
                df_scraped.to_csv(history_path, index=False)

            # Still save latest snapshot (your app depends on this)
            df_scraped.to_csv("data/latest_pokemon_prices.csv", index=False)

            st.success("Data refreshed!")
            df = df_scraped  # ✅ assign top-level, do not return
        else:
            st.error("Scraping failed or returned no data.")
            st.stop()

# --------------------------
# Load existing data if not refreshed
@st.cache_data
def get_valid_data():
    df = load_data()  # Your load_data() function

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

# Only load from saved file if button hasn't just refreshed
if "df" not in locals():  # avoid overwriting freshly scraped df
    df = get_valid_data()

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

# --------------------------
# Filter Controls
# --------------------------
st.sidebar.header("Filter Options")

all_sets = sorted(df["Set"].unique())
select_all_sets = st.sidebar.checkbox("Select all sets", value=True)

if select_all_sets:
    selected_sets = all_sets
else:
    selected_sets = st.sidebar.multiselect("Choose Set(s)", options=all_sets)

if not selected_sets:
    st.warning("Please select at least one set.")
    st.stop()

min_ungraded = st.sidebar.number_input("Min price ($)", min_value=0.01, max_value=10000.0, value=0.01, step=0.01)
max_ungraded = st.sidebar.number_input("Max price ($)", min_value=0.01, max_value=10000.0, value=50.0, step=0.01)
min_grade9 = st.sidebar.number_input("Min Grade 9 Price", min_value=0, value=0)
min_psa10 = st.sidebar.number_input("Min PSA 10 Price", min_value=0, value=0)

# --------------------------
# 3-day, 7-day, 14-day, and 30-day Ungraded Price Change Filters
change_filters = {}

for days in [3, 7, 14, 30]: 
    col_name = f"Ungraded_{days}d_Change"
    if col_name in df.columns:
        min_val, max_val = st.sidebar.slider(
            f"{days}-Day Ungraded Price Change ($)",
            min_value=-100.0,
            max_value=100.0,
            value=(-100.0, 100.0),
            step=0.01
        )
        change_filters[col_name] = (min_val, max_val)
    else:
        change_filters[col_name] = (-100.0, 100.0)

# --------------------------
# Apply all filters
filtered = df[
    (df["Set"].isin(selected_sets)) &
    (df["Ungraded_Price"].between(min_ungraded, max_ungraded)) &
    (df["Grade_9_Price"] >= min_grade9) &
    (df["PSA_10_Price"] >= min_psa10)
]

# Apply 3d, 7d, 14d change filters if columns exist
for col_name, (min_val, max_val) in change_filters.items():
    if col_name in filtered.columns:
        filtered = filtered[filtered[col_name].between(min_val, max_val)]

st.subheader(f"Filtered Results ({len(filtered)} cards)")

# Add a sidebar checkbox to control image column visibility
# Warn users before enabling thumbnails
st.sidebar.caption("⚠️ Enabling image thumbnails may slow down loading or cause images to fail.")
show_images = st.sidebar.checkbox("Show image thumbnails", value=False)

if show_images:
    # Create image column with HTML <img> tag wrapped in a link
    def image_tag(url):
        return f'<a href="{url}" target="_blank"><img src="{url}" width="80"></a>'

    styled_df = filtered.copy()
    styled_df["Image"] = styled_df["Image_URL"].apply(image_tag)
    styled_df = styled_df.drop(columns=["Image_URL"])  

    # Reorder so image appears first (optional)
    cols = ["Image"] + [col for col in styled_df.columns if col != "Image"]
    styled_df = styled_df[cols]

    st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    filtered_display = filtered.drop(columns=["Image_URL"])
    st.dataframe(filtered_display.reset_index(drop=True))


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

# Load full history
history_path = "data/pokemon_price_history.csv"
if os.path.exists(history_path):
    history_df = pd.read_csv(history_path)
else:
    st.warning("No history file found.")
    history_df = pd.DataFrame()

# Work with full history + keep latest snapshot separate
if not history_df.empty:
    history_df["Date"] = pd.to_datetime(history_df["Date"])
    latest_date = history_df["Date"].max()
    latest_df = history_df[history_df["Date"] == latest_date].copy()
else:
    latest_df = pd.DataFrame()

# Use latest_df for the main app display
df = latest_df.copy()

# Compute 3, 7, 14, 30 day price changes inside the full history
if not history_df.empty:
    for days in [3, 7, 14, 30]:
        prior_cutoff = history_df["Date"].max() - pd.Timedelta(days=days)
        prior = history_df[history_df["Date"] <= prior_cutoff]

        if not prior.empty:
            prior_prices = (
                prior.groupby("Card_Name")
                .apply(lambda x: x.sort_values("Date").iloc[-1])
                .reset_index(drop=True)
            )

            history_df = pd.merge(
                history_df,
                prior_prices[["Card_Name", "Ungraded_Price"]],
                on="Card_Name",
                how="left",
                suffixes=("", f"_{days}d_ago")
            )

            history_df[f"Ungraded_{days}d_Change"] = (
                history_df["Ungraded_Price"] - history_df[f"Ungraded_Price_{days}d_ago"]
            )

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
# --- GenAI Chatbot Section ---
st.markdown("---")
st.subheader("GenAI Chatbot")

# Initialize Hugging Face LLM & Pandas Agent
llm = HFInferenceLLM(
    model_name="HuggingFaceTB/SmolLM3-3B",
    api_key=os.environ["HF_TOKEN"]
)

agent = create_pandas_dataframe_agent(
    llm,
    df,  # your current all-history dataframe
    verbose=False,
    allow_dangerous_code=True
)

# Session state to keep chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input for chat
user_input = st.text_input("Ask a question about the dataset:")

if st.button("Ask"):
    if user_input:
        with st.spinner("Thinking..."):
            try:
                answer = agent.run(user_input)
                st.session_state.chat_history.append({"user": user_input, "bot": answer})
            except Exception as e:
                st.session_state.chat_history.append({"user": user_input, "bot": f"⚠️ Error: {e}"})

# Display chat history
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown("---")

st.markdown("Built by Logan Laszewski")
