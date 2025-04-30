Capital One Airline Route Profitability Analysis

This project was built for the Capital One Data Challenge. The goal was to identify the top 5 most profitable airline routes using flight-level data, ticket sales, and airport info. I focused on building a clean, modular, and reproducible pipeline with solid data quality checks.

Objective

Recommend 5 round-trip airline routes (between medium and large U.S. airports) that maximize profitability while maintaining high on-time performance.

📁 Project Structure
capital_one_data_challenge/
│
├── data/
│ ├── raw/ <- Original CSVs provided (Flights, Tickets, Airport Codes)
│ ├── meta_data/ <- Metadata file from Capital One
│ └── cleaned/ <- Cleaned datasets after QC
│
├── notebooks/
│ ├── run_full_qc.ipynb <- Cleans and validates raw data
│ ├── eda_medium_large.ipynb <- Performs filtered EDA (medium ↔ large airport routes)
│ └── final_answers.ipynb <- Route selection, KPIs, and visualizations
│
├── src/
│ ├── data_cleaner.py <- Data cleaning and formatting functions
│ ├── merge_utils.py <- Join logic for flights, tickets, airports
│ └── plot_utils.py <- Plotly-based custom visualizations
│
├── requirements.txt
└── README.md

How to Run
Make sure your working directory is the root of capital_one_data_challenge.
Step 1: Install dependencies  
Install Dependencies
(I recommend using a virtual environment)
pip install -r requirements.txt

Step 2: Run notebooks in this order:
Note: Before running make sure the 3 datasets(Flights.csv, Tickets.csv, Airport_Codes.csv) are placed in the data/raw/ folder.
run_full_qc.ipynb
aggregation_logic.ipynb
eda_medium_large.ipynb
final_answers.ipynb

KPIs Computed
These are detailed in a separate document, but in short — I computed estimated passengers, ticket revenue, baggage revenue, total revenue, total cost, and final profit per route.

Source of Data
All datasets were provided by Capital One as part of their Airline Route Optimization Challenge.

Alignment with Capital One Evaluation Criteria

1. Builder Mindset
   I approached this challenge with a focus on modular, reusable, and scalable code design.

Modular Codebase with Reusability
Created reusable functions for cleaning, merging, and plotting within src/:

data_cleaner.py handles type conversion, formatting inconsistencies, outlier detection/capping, and imputation.

merge_utils.py includes multiple scalable join strategies using Pandas, Dask, and Polars, allowing the project to adapt to datasets of varying size and memory footprint.

plot_utils.py includes advanced and reusable Plotly visualizations to support interpretability and storytelling.

The notebooks are slim and call these utility functions instead of embedding complex logic inline.

Effective Formatting, Commenting & Conciseness
Each script is written with clear structure:

Top-level constants for config

Well-named functions with docstrings

In-notebook comments for clarity

Proper use of logging-style print() statements to indicate process stages

Functional + Adaptive Design
Code uses parameterized helper functions (like multi_strategy_imputer(), batch_plot_outliers(), and analyze_conversion_errors()) to apply logic flexibly across columns.

Joins and aggregations are written once and reused across Pandas/Dask/Polars to evaluate scalability under memory pressure.

2. Data Management
   Systematic Data Quality Checks
   Comprehensive QC performed in run_full_qc.ipynb, including:

Null pattern analysis (via missingno)

Column-wise type conversion with detailed error diagnostics

Text-to-numeric normalization for messy input values (e.g. “Thirty Five” → 35)

Outlier detection using IQR and MAD methods

Abnormality flags for columns expected to be numeric/dates

Documentation of Issues
Conversion issues were tracked per column with a structured error report:

Type of issue (e.g., invalid chars, parsing failure)

Affected value examples

This was implemented in analyze_conversion_errors() which returns:

A report summary

A clean dataframe

A list of error cases for debugging

Deliberate Resolution Steps
Missing values handled via:

Statistical imputation (mean, median) for numerical

“Unknown” or model-based fill for categorical

Columns with >99% nulls dropped

Cleaned datasets saved to /data/cleaned/ for reproducibility

Metadata for Derived Columns
Each newly engineered column is clearly named and documented:

3. Reproducible Pipeline using Modular Python Scripts
   Data pipeline runs in three defined stages:

run_full_qc.ipynb → cleans raw data using data_cleaner.py

eda_medium_large.ipynb → filters for relevant airports and runs profiling/EDA

final_answers.ipynb → computes KPIs and selects routes

Every transformation is:

Modularized in a function

Logged clearly in outputs

Re-run-safe, thanks to consistent naming and saved CSVs

4. Custom Merge Logic (Tested across Libraries)
   To simulate real-world scaling needs, the dataset merging logic was written generically and tested with:

Method File
Pandas (default) merge_flights_tickets_airports_pandas()
Dask (parallel processing) merge_flights_tickets_airports_dask()
Polars (high-performance) merge_flights_tickets_airports_polars()
All approaches:

Merge flights and tickets on ROUTE_KEY + carrier

Split ROUTE_KEY into AIRPORT_A and AIRPORT_B

Join airport metadata for both sides

Produce a final enriched dataset ready for KPI computation
