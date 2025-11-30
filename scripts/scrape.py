"""
Google Play Store Review Scraper
Task 1: Data Collection

Scrapes reviews for Ethiopian bank mobile apps using google-play-scraper.
Target: 400+ reviews per bank.
"""

import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google_play_scraper import app, Sort, reviews
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
# from config import APP_IDS, BANK_NAMES, SCRAPING_CONFIG, DATA_PATHS
# Try package-style importing, fall back to top-level config
try:
    from scripts.config import APP_IDS, BANK_NAMES, SCRAPING_CONFIG, DATA_PATHS
except ModuleNotFoundError:
    from config import APP_IDS, BANK_NAMES, SCRAPING_CONFIG, DATA_PATHS


class PlayStoreScraper:
    """Scraper class for gathering Google Play Store reviews."""

    def __init__(self):
        # Load configuration variables from the config file
        self.app_ids = APP_IDS
        self.bank_names = BANK_NAMES
        self.reviews_per_bank = SCRAPING_CONFIG["reviews_per_bank"]
        self.lang = SCRAPING_CONFIG["lang"]
        self.country = SCRAPING_CONFIG["country"]
        self.max_retries = SCRAPING_CONFIG["max_retries"]

    def get_app_info(self, app_id: str):
        """Fetch metadata for the app (title, score, installs, etc.)."""
        try:
            info = app(app_id, lang=self.lang, country=self.country)
            return {
                "app_id": app_id,
                "title": info.get("title", "N/A"),
                "score": info.get("score", 0),
                "ratings": info.get("ratings", 0),
                "reviews": info.get("reviews", 0),
                "installs": info.get("installs", "N/A"),
            }
        except Exception as e:
            print(f"Error fetching app info for {app_id}: {e}")
            return None

    def scrape_reviews(self, app_id: str, count: int):
        """Scrape N reviews for a specific app with retry logic."""
        for attempt in range(self.max_retries):
            try:
                data, _ = reviews(
                    app_id,
                    lang=self.lang,
                    country=self.country,
                    sort=Sort.NEWEST,
                    count=count,
                    filter_score_with=None,
                )
                print(f"Successfully scraped {len(data)} reviews")
                return data
            except Exception as e:
                print(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Failed to scrape reviews for {app_id}")
                    return []
        return []

    def process_reviews(self, reviews_data, bank_code):
        """Extract only required fields and attach bank metadata."""
        processed = []

        for r in reviews_data:
            processed.append({
                "review_id": r.get("reviewId", ""),
                "review_text": r.get("content", ""),
                "rating": r.get("score", 0),
                "review_date": r.get("at", datetime.now()),
                "user_name": r.get("userName", "Anonymous"),
                "thumbs_up": r.get("thumbsUpCount", 0),
                "reply_content": r.get("replyContent", None),
                "bank_code": bank_code,
                "bank_name": self.bank_names[bank_code],
                "app_id": r.get("reviewCreatedVersion", "N/A"),
                "source": "Google Play",
            })

        return processed

    def scrape_all_banks(self):
        """Run full scraping pipeline across all banks."""
        all_reviews = []
        app_info_records = []

        print("=" * 60)
        print("STARTING GOOGLE PLAY STORE SCRAPER")
        print("=" * 60)

        # ---- Phase 1: Fetch App Metadata ----
        print("\n[1/2] Fetching app info...")
        for code, app_id in self.app_ids.items():
            print(f"\n{code}: {self.bank_names[code]}")
            print(f"App ID: {app_id}")

            info = self.get_app_info(app_id)
            if info:
                info["bank_code"] = code
                info["bank_name"] = self.bank_names[code]
                app_info_records.append(info)
                print(f"    Current Rating: {info['score']}")
                print(f"    Total Ratings: {info['ratings']}")
                print(f"    Total Reviews: {info['reviews']}")

        # Save app info
        if app_info_records:
            df_info = pd.DataFrame(app_info_records)
            os.makedirs(DATA_PATHS["raw"], exist_ok=True)
            df_info.to_csv(f"{DATA_PATHS['app_info']}", index=False)
            print(f"\nApp information saved to {DATA_PATHS['app_info']}")

        # ---- Phase 2: Scrape Reviews ----
        print("\n[2/2] Scraping reviews...")
        for code, app_id in tqdm(self.app_ids.items(), desc="Banks"):
            raw_reviews = self.scrape_reviews(app_id, self.reviews_per_bank)
            if raw_reviews:
                formatted = self.process_reviews(raw_reviews, code)
                all_reviews.extend(formatted)
                print(f"Collected {len(formatted)} reviews for {self.bank_names[code]}")
            else:
                print(f"WARNIGN: No reviews collected for {self.bank_names[code]}")
        time.sleep(2)

        # --- Phase 3: Save all reviews ---
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            os.makedirs(DATA_PATHS["raw"], exist_ok=True)
            df.to_csv(DATA_PATHS["raw_reviews"], index=False)

            print("=" * 60)
            print("SCRAPING COMPLETE")
            print("=" * 60)
            print(f"Total reviews collected: {len(df)}")

            print(f"Reviews per bank:")
            for bank_code in self.bank_names.keys():
                count = len(df[df['bank_code'] == bank_code])
                print(f"  {self.bank_names[bank_code]}: {count}")

            print(f"\nData saved to: {DATA_PATHS['raw_reviews']}")

            return df
        else:
            print("\nERROR: No reviews were collected!")
            return pd.DataFrame()

    def display_sample_reviews(self, df, n=3):
        """
        Display sample reviews from each bank to verify data quality.
        """
        print("\n" + "=" * 60)
        print("SAMPLE REVIEWS")
        print("=" * 60)

        for bank_code in self.bank_names.keys():
            bank_df = df[df['bank_code'] == bank_code]
            if not bank_df.empty:
                print(f"\n{self.bank_names[bank_code]}:")
                print("-" * 60)
                samples = bank_df.head(n)
                for idx, row in samples.iterrows():
                    print(f"\nRating: {'⭐' * row['rating']}")
                    print(f"Review: {row['review_text'][:200]}...")
                    print(f"Date: {row['review_date']}")
                    
def main():
    """Main execution function"""
    scraper = PlayStoreScraper()
    df = scraper.scrape_all_banks()
    if not df.empty:
        scraper.display_sample_reviews(df)

    return df

if __name__ == "__main__":
    review_df = main()
