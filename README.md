# 📊 Fintech Mobile App Review Analysis — 10 Academy Week 2

This project analyzes customer experiences of major Ethiopian banking mobile apps by scraping, preprocessing, analyzing, and visualizing Google Play Store reviews.  
It completes **Task 1 (Scraping & Preprocessing)** and **Task 2 (EDA)** of the Week 2 challenge.

## ⭐ Project Overview

Mobile banking apps are central to Ethiopia’s digital financial ecosystem. Understanding user sentiment, key frustrations, and UX gaps can meaningfully improve product decisions.

This project builds a **data pipeline** that:

1. **Scrapes reviews** from the Google Play Store  
2. **Preprocesses and cleans raw user feedback**  
3. **Conducts exploratory data analysis (EDA)**  
4. Performs **sentiment analysis**, **topic modeling**, and **insight reporting**

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
├── data/
│   ├── processed/
│   └── raw/
│
├── notebooks/
│   ├── __init__.py
│   ├── preprocessing_EDA.ipynb
│   └── README.md
│
├── plots/
│
├── scripts/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── README.md
│   └── scrape.py
│
├── src/
│   └── __init__.py
│
├── tests/
│   └── __init__.py
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

# 📥 Task 1: Scraping

### Run scraper:
```bash
python scripts/scrape.py
```

Outputs:
- `data/raw/app_info.csv`
- `data/raw/reviews_raw.csv`

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

# 🛠️ Technologies Used

| Category       | Tools |
|----------------|-------|
| Scraping       | google-play-scraper, tqdm |
| Preprocessing  | pandas, numpy, regex |
| Environment    | python-dotenv |
| Visualization  | matplotlib, pandas |
| Notebook       | Jupyter Notebooks |
