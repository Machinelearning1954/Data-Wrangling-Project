# Machine Learning Engineering Bootcamp Capstone: Data Wrangling

## Project Overview

This project focuses on the data wrangling phase of a machine learning project. The primary goal is to collect data from multiple disparate sources, clean and merge this data, perform feature engineering, and visualize the data to gain insights and guide further steps. The process emphasizes thoughtful decision-making regarding missing values, outliers, and the overall preparation of a dataset suitable for machine learning modeling.

This work fulfills Step 5 of the Machine Learning Engineering Bootcamp Capstone, focusing on data wrangling techniques.

## Data Sources

1.  **Apple (AAPL) Stock Data:**
    *   **Source:** Yahoo Finance API (`YahooFinance/get_stock_chart` via a data API provider).
    *   **Description:** Daily historical stock data for Apple Inc. (AAPL) covering a 5-year period. Includes Open, High, Low, Close (OHLC) prices, trading volume, and adjusted close prices.
    *   **File:** `/data/aapl_stock_data.json` (raw data)

2.  **US Gross Domestic Product (GDP) Data:**
    *   **Source:** World Bank Data Portal (CSV download).
    *   **Description:** Annual Nominal Gross Domestic Product (current US$) for the United States. Indicator code: `NY.GDP.MKTP.CD`.
    *   **Rationale for Change:** Initially, an attempt was made to fetch this data via the World Bank API (`DataBank/indicator_data`). However, an API authentication failure necessitated a switch to downloading the data directly as a CSV from the World Bank website.
    *   **File:** `/data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_85078.csv` (raw data from World Bank ZIP)

## Project Structure

```
/
|-- data/
|   |-- aapl_stock_data.json
|   |-- API_NY.GDP.MKTP.CD_DS2_en_csv_v2_85078.csv
|   |-- gdp_data.zip (original downloaded zip for GDP)
|   |-- us_gdp_data.json (initial attempt, failed API call artifact)
|   |-- aapl_gdp_merged_cleaned.csv (intermediate merged data)
|   |-- aapl_gdp_wrangled_features.csv (final dataset with engineered features)
|-- notebooks/
|   |-- data_wrangling.ipynb
|   |-- aapl_adjclose_ma.png
|   |-- aapl_volume.png
|   |-- aapl_daily_return.png
|   |-- aapl_volatility.png
|   |-- us_gdp_time_series.png
|   |-- us_gdp_growth_rate.png
|   |-- aapl_daily_return_distribution.png
|   |-- aapl_adjclose_distribution.png
|   |-- correlation_matrix.png
|-- README.md
|-- requirements.txt
|-- todo.md (internal checklist used during development)
|-- fetch_data.py (script for initial data fetching attempt)
```

## Data Wrangling Process

The detailed data wrangling process is documented step-by-step in the Jupyter Notebook: [`notebooks/data_wrangling.ipynb`](notebooks/data_wrangling.ipynb).

The key steps include:

1.  **Data Collection:** Fetching AAPL stock data via API and downloading US GDP data as a CSV.
2.  **Data Cleaning:**
    *   Handling missing values in both datasets (e.g., ffill/bfill for stock prices, dropping rows for GDP where necessary).
    *   Addressing data types and ensuring consistency.
    *   Reshaping the GDP data from wide to long format.
3.  **Data Merging:** Combining the daily stock data with annual GDP data based on the year.
4.  **Feature Engineering:** Creating new features such as:
    *   Daily stock returns.
    *   Moving averages (7-day and 30-day) for adjusted close price.
    *   30-day rolling volatility of daily returns.
    *   Year-over-Year GDP growth rate.
    *   Lagged stock features and date-based features (month, day of week).
5.  **Data Visualization:** Creating various plots (time series, distributions, heatmaps) to understand the data, the impact of wrangling steps, and to guide decisions.
6.  **Validation:** Reviewing the entire process, justifying decisions, and assessing the quality and suitability of the final dataset for ML modeling.

## How to Run

1.  **Environment Setup:**
    *   It is recommended to use a Python environment (e.g., venv or conda).
    *   Install the required libraries using the `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```

2.  **Jupyter Notebook:**
    *   Open and run the Jupyter Notebook `notebooks/data_wrangling.ipynb` to see the entire data wrangling process, from data loading to visualization and final dataset creation.
    *   The notebook loads the raw data from the `/data` directory and saves intermediate and final processed files back to this directory.
    *   All visualizations are generated and saved to the `/notebooks` directory by the notebook.

## Key Decisions and Justifications

*   **Handling Missing Stock Data:** Forward-fill and backward-fill were used for price data, a common practice for financial time series. Volume NaNs were filled with 0.
*   **Handling Missing GDP Data:** Years with missing GDP values in the World Bank dataset were dropped after reshaping to ensure data accuracy.
*   **Outlier Treatment:** No explicit outlier removal was performed on stock prices, as large movements can be legitimate. Focus was on data integrity from source and handling missing values.
*   **Feature Engineering Choices:** Features like moving averages, volatility, and returns are standard in financial analysis. GDP growth YoY provides macroeconomic context.
*   **Merging Strategy:** A left merge of daily stock data with annual GDP data (on 'Year') was chosen to retain all stock records while adding annual economic context.

## Results

The primary output of this data wrangling phase is the cleaned, merged, and feature-enriched dataset: `/data/aapl_gdp_wrangled_features.csv`.

This dataset is now better prepared for subsequent machine learning tasks, such as time series forecasting or analysis of stock performance in relation to economic indicators.

