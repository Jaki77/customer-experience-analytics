"""
Configuration file for Bank Reviews Analysis Project
Target: CBE, Bank of Abyssinia, Dashen Bank
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OFFICIAL APPS AND THIER ID (Updated Nov 2025)
APP_IDS = {
    'CBE': 'com.combanketh.mobilebanking',         # CBE Mobile Banking App
    'BOA': 'com.boa.boaMobileBanking',             # BOA's Mobile Banking App
    'Dashen': 'com.dashen.dashensuperapp'          # Dashen Mobile Banking App
}

# Full bank names for reporting
BANK_NAMES = {
    'CBE': 'Commercial Bank of Ethiopia',
    'BOA': 'Bank of Abyssinia',
    'Dashen': 'Dashen Bank'
}

# Scraping settings
SCRAPING_CONFIG = {
    'reviews_per_bank': int(os.getenv('REVIEWS_PER_BANK', 500)),  
    'max_retries': int(os.getenv('MAX_RETRIES', 3)),
    'lang': 'en',
    'country': 'et'  # Ethiopia
}

# Database configuartions
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'bank_reviews'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
    'schema': 'public'
}

# File paths
DATA_PATHS = {
    'raw': '../data/raw',
    'processed': '../data/processed',
    'analysis': '../data/processed/analysis',
    'database': '../data/database',
    'raw_reviews': '../data/raw/reviews_raw.csv',
    'processed_reviews': '../data/processed/reviews_processed.csv',
    'app_info': '../data/raw/app_info.csv',
    'sentiment_results': '../data/processed/analysis/sentiment_results.csv',
    'thematic_results': '../data/processed/analysis/themes_summary.csv',
    'final_analysis': '../data/processed/analysis/final_analysis_results.csv'
}

# Create directories if not exist
for path in [DATA_PATHS['raw'], DATA_PATHS['processed'], DATA_PATHS['analysis'], DATA_PATHS['database']]:
    os.makedirs(path, exist_ok=True)