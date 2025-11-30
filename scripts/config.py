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

# File paths
DATA_PATHS = {
    'raw': '../data/raw',
    'processed': '../data/processed',
    'raw_reviews': '../data/raw/reviews_raw.csv',
    'processed_reviews': '../data/processed/reviews_processed.csv',
    'app_info': '../data/raw/app_info.csv'
}

# Create directories if not exist
os.makedirs(DATA_PATHS['raw'], exist_ok=True)
os.makedirs(DATA_PATHS['processed'], exist_ok=True)