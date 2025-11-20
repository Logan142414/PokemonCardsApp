# Pokémon Price Tracker

**Author:** Logan Laszewski
**Project Type:** Web Scraping, Data Analysis, Streamlit App
**Last Updated:** November 2025

---

## Overview

The Pokémon Price Tracker is a Python web scraping project designed to track Pokémon card prices and analyze market trends over time. The project consists of two main parts:

1. **Real time price scraping and filtering (Part 1)**
2. **Historical tracking, cloud storage, and interactive analysis (Part 2)**

---

## Features

### Part 1: Web Scraping & Streamlit App

* Scrapes Pokémon card names and prices from [PriceCharting.com](https://www.pricecharting.com/).
* Extracts ungraded, Grade 9, and PSA 10 prices.
* Interactive Streamlit app with:
  * One-click dataset refresh
  * Filtering by price, set, or grading
  * “Deal value” calculation (Grade 9 price minus ungraded price)
  * Card image previews
* Learned techniques:

  * HTML inspection and parsing with BeautifulSoup
  * Requests for automated data retrieval
  * Handling partially loaded data

### Part 2: Historical Tracking & Analysis

* Persistent cloud storage using Google Cloud Storage to store historical data.
* Tracks daily snapshots of card prices.
* Calculates price changes over 3, 7, 14, and 30-day periods.
* Visualizations:

  * Average price changes over time (bar chart)
  * Ungraded price trends (line chart)
* CSV export for offline analysis.
* Prioritizes scraping high-value cards using URL sorting by price.
* Optional generative AI integration for natural language insights (experimented using LangChain and FAISS embeddings).

---

## Technical Details

**Languages & Libraries:**

* Python, Pandas, NumPy
* BeautifulSoup, Requests
* Streamlit for interactive dashboards
* Google Cloud Storage API for cloud persistence
* Optional: Selenium/Playwright for dynamic content (future enhancement)

**Scraping Approach:**

1. Inspect HTML structure of card sets and individual cards.
2. Extract relevant `<table>` and `<td>` elements.
3. Use Python loops to iterate over sets and append data to a dataset.
4. For historical tracking, append daily snapshots to a cumulative CSV in cloud storage.

**Price Change Calculations:**

* Compare current prices to snapshots from 3, 7, 14, and 30 days ago.
* Detect trends and highlight potential investment opportunities.

---

## Project Learnings

* Inspecting HTML and understanding web structure is crucial for effective scraping.
* Dynamic JavaScript content requires advanced tools like Selenium or Playwright.
* APIs are often a more reliable alternative to scraping.
* Automating historical tracking and cloud storage ensures data persistence and trend analysis.
* Interactive dashboards make large datasets usable for non-technical users.

---

## Future Enhancements

* Full dataset capture using browser automation tools.
* Automate daily scraping with GitHub Actions or Cloud Functions.
* Explore more advanced AI-driven insights for price trends and investment suggestions.

---

## Pokémon Card Pricing App
- Streamlit demo: [pokemoncards-exploration.streamlit.app](https://pokemoncards-exploration.streamlit.app/)

---

## Medium Articles
- Part 1: [Building a Pokémon Card Price Tracker App with Web Scraping](https://medium.com/@logan.laszewski14/learning-web-scraping-by-tracking-pokémon-card-prices-30d97a8f5eeb)
- Part 2: [Pokémon Price Tracker Part 2: Historical Tracking, Cloud Storage, and Interactive Analysis](https://medium.com/@logan.laszewski14/pok%C3%A9mon-price-tracker-part-2-historical-tracking-cloud-storage-and-interactive-analysis-af61567ee335)
