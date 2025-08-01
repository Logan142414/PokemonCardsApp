import streamlit as st
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
from datetime import datetime
from zoneinfo import ZoneInfo

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

    /* Input fields text color */
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
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.image("pikachu-running.png", use_container_width = True)

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


# Show refresh button
if st.button("Refresh Price Data"):
    with st.spinner("Scraping all Pokémon card sets (this may take up to 5 minutes)..."):
        df_scraped = scrape_pricecharting_data()

        if not df_scraped.empty:
            os.makedirs("data", exist_ok=True)
            df_scraped.to_csv("data/latest_pokemon_prices.csv", index=False)
            st.success("Data refreshed!")
            df = df_scraped  # ✅ use the scraped data immediately
        else:
            st.error("Scraping failed or returned no data.")
            st.stop()

# ✅ Always load from saved file if not scraped
@st.cache_data
def get_valid_data():
    df = load_data()

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

df = get_valid_data()

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
