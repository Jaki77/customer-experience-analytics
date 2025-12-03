# 📊 Fintech Mobile App Review Analysis — 10 Academy Week 2

This project analyzes customer experiences of major Ethiopian banking mobile apps by scraping, preprocessing, analyzing, and visualizing Google Play Store reviews.  

## ⭐ Project Overview

Mobile banking apps are central to Ethiopia’s digital financial ecosystem. Understanding user sentiment, key frustrations, and UX gaps can meaningfully improve product decisions.

This project builds a **data pipeline** that:

1. **Scrapes reviews** from the Google Play Store  
2. **Preprocesses and cleans raw user feedback**  
3. **Conducts exploratory data analysis (EDA)**  
4. Performs **sentiment analysis**, **topic modeling**, and **insight reporting**
5. **Identify drivers** and pain points in mobile banking apps
6. **Store cleaned data** in PostgreSQL database
7. **Deliver actionable insights** with visualizations and recommendations

## 🏦 Banking Apps Analyzed

| Bank Code | App Name                       | App ID                                      |
|----------|---------------------------------|---------------------------------------------|
| CBE      | Commercial Bank of Ethiopia     | `com.combanketh.mobilebanking`              |
| BOA    | Bank of Abyssinia           | `com.boa.boaMobileBanking`                           |
| Dashen | Dashen Bank                   | `com.dashen.dashensuperapp`         |

App IDs are read dynamically from `.env`.

## 🧱 Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── unittest.yml
│
├── .vscode/
│   └── settings.json
│
├── data/                       # Data storage
│   ├── raw/                   # Raw scraped data
│   ├── processed/             # Cleaned and processed data
│   ├── analysis/              # Analysis results
│   └── database/              # Database documentation
│
├── notebooks/
│   ├── __init__.py
│   ├── preprocessing_EDA.ipynb
│   ├── visualization.ipynb
│   └── README.md
│
├── plots/
│
├── scripts/
│   ├── config.py              # Configuration and settings
│   ├── scrape.py              # Web scraping (Task 1)
│   ├── preprocess.py          # Data cleaning (Task 1)
│   ├── sentiment.py           # Sentiment analysis (Task 2)
│   ├── themes.py              # Thematic analysis (Task 2)
│   ├── analysis_pipeline.py   # Orchestrates Task 2 analysis
│   ├── database.py            # Database operations (Task 3)
│   ├── db_schema.sql          # PostgreSQL schema definition
│   └── db_run.py           # Simplified Task 3 runner
│
├── src/
│   └── __init__.py
│
├── tests/
│   ├── __init__.py
│   └── database_test.py    # Test database functionality
│
├── .gitignore
├── LICENSE
├── README.md          
└── requirements.txt
```

## ⚙️ Setup Instructions

### 1. Clone repository
```bash
git clone https://github.com/Jaki77/customer-experience-analytics.git
cd customer-experience-analytics
```

### 2. Create & activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add `.env` file
Create a `.env` in the project root:
```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=bank_reviews
DB_USER=postgres
DB_PASSWORD=your_password

# Scraping Configuration
REVIEWS_PER_BANK=500
MAX_RETRIES=3
```
# 📥 Task 1: Scraping

### Run scraper:
```bash
python scripts/scrape.py
```

Outputs:
- `data/raw/app_info.csv`
- `data/raw/reviews_raw.csv`
- `data/processed/reviews_processed.csv`
- `data/processed/analysis/sentiment_results.csv`
- `data/processed/analysis/themes_summary.csv`
- `fdata/processed/analysis/final_analysis_results.csv`


# 🧹 Task 1: Preprocessing

### Run preprocessor:
```bash
python scripts/preprocess.py
```

Output:
- `data/processed/reviews_processed.csv`

# 📊 Task 2: Exploratory Data Analysis (EDA)

The EDA focuses on:
- Rating distribution  
- Reviews per bank  
- Text length distribution  
- Time-based review trends  
- Common words/phrases  

Notebooks:
- `notebooks/preprocessing_EDA.ipynb`

# 📊 Task 3: Store Cleaned Data in PostgreSQL
```bash
# Run complete database pipeline
python scripts/database.py

# Or run step by step
python scripts/database.py --create-only    # Create database and tables
python scripts/database.py --skip-insert    # Skip data insertion
python scripts/database.py --verify-only    # Verify existing data

# Test database functionality
python scripts/test_database.py
```
Output:
- `data/database/schema_dump.sql`

# Database Schema
```bash
-- Main Tables
banks              -- Bank information
reviews            -- Review data with sentiment
themes             -- Theme categories
review_themes      -- Many-to-many relationship

-- Analytical Views
bank_summary              -- Bank performance metrics
monthly_sentiment_trends  -- Time-based analysis
theme_analysis           -- Theme frequency by bank
rating_distribution      -- Rating statistics
```

# 🛠️ Technologies Used

| Category       | Tools |
|----------------|-------|
| Scraping       | google-play-scraper, tqdm |
| Preprocessing  | pandas, numpy, regex |
| Environment    | python-dotenv |
| Visualization  | matplotlib, pandas |
| Notebook       | Jupyter Notebooks |
| NLP Libraries       | Transformers, NLTK, spaCy, TextBlob, VADER |
| Database       | PostgreSQL, psycopg2 |
| Environment Management      | python-dotenv |