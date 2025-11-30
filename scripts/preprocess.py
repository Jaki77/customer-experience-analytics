"""
Data Preprocessing Script
Task 1: Cleaning and standardizing scraped reviews.
"""

import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import re
from datetime import datetime
# from config import DATA_PATHS
# Try package-style importing, fall back to top-level config
try:
    from scripts.config import DATA_PATHS
except ModuleNotFoundError:
    from config import DATA_PATHS

class ReviewPreprocessor:
    """Preprocess Google Play Store review dataset."""

    def __init__(self, input_path=None, output_path=None):
        """
        Initialize preprocessor

        Args:
            input_path (str): Path to raw reviews CSV
            output_path (str): Path to save processed reviews
        """
        self.input_path = input_path or DATA_PATHS["raw_reviews"]
        self.output_path = output_path or DATA_PATHS["processed_reviews"]
        self.df = None
        self.stats = {}

    # --------------------------------------------------------

    def load_data(self):
        """Load raw reviews data"""
        print("Loading raw data...")

        try:
            self.df = pd.read_csv(self.input_path)
            self.stats["original_count"] = len(self.df)
            print(f"Loaded {len(self.df)} raw reviews")
            return True
        except FileExistsError:
            print(f"ERROR: File no found: {self.input_path}")
            return False
        except Exception as e:
            print(f"ERROR: Failed to load data: {e}")
            return False

    # --------------------------------------------------------

    def check_missing(self):
        """Check for missing data"""
        print("\n[1/7] Checking missing values...")
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100

        print("\nMissing values:")
        for col in missing.index:
            if missing[col] > 0:
                print(f" {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
        self.stats["missing_before"] = missing.to_dict()

        # Define columns necessary for the analysis
        critical_cols = ['review_text', 'rating', 'bank_name']
        missing_critical = self.df[critical_cols].isnull().sum()

        if missing_critical.sum() > 0:
            print("\nWARNING: Missing values in critical columns: ")
            print(missing_critical[missing_critical > 0])

    # --------------------------------------------------------

    def remove_duplicates(self):
        """Remove duplicate reviews by ID and text."""
        print("\n[2/7] Removing duplicate...")

        before = len(self.df)
        # Remove duplicates based on review_id
        if "review_id" in self.df.columns:
            self.df = self.df.drop_duplicates(subset=["review_id"])
        
        # Remove duplicates based on review_text as fallback
        self.df = self.df.drop_duplicates(subset=["review_text"])

        removed = before - len(self.df)
        self.stats["duplicates_removed"] = removed
        self.stats["count_after_duplicates"] = len(self.df)

        if removed > 0:
            print(f"Removed {removed} duplicate reviews")

    # --------------------------------------------------------

    def handle_missing(self):
        """Handle missing values"""
        print("\n[3/7] Handling missing critical data...")
        critical_cols = ["review_text", "rating", "bank_name"]

        before = len(self.df)
        self.df = self.df.dropna(subset=critical_cols)
        removed = before - len(self.df)
        self.stats["removed_missing"] = removed

        if removed > 0:
            print(f"Removed {removed} rows with missing critical values")

        self.df["user_name"] = self.df["user_name"].fillna("Anonymous")
        self.df["thumbs_up"] = self.df["thumbs_up"].fillna(0)
        self.df["reply_content"] = self.df["reply_content"].fillna("")

        self.stats["rows_removed_missing"] = removed
        self.stats["count_after_missing"] = len(self.df)

    # --------------------------------------------------------

    def normalize_dates(self):
        """Normalize date formats to YYYY-MM-DD"""
        print("\n[4/7] Normalizing dates...")

        try:
            self.df["review_date"] = pd.to_datetime(self.df["review_date"]).dt.date
            self.df["review_year"] = pd.to_datetime(self.df["review_date"]).dt.year
            self.df["review_month"] = pd.to_datetime(self.df["review_date"]).dt.month

            print(f"Date range: {self.df['review_date'].min()} to {self.df['review_date'].max()}")

        except Exception as e:
            print(f"WARNING: Error normalizing dates: {(e)}")

    # --------------------------------------------------------

    def clean_text(self):
        """Clean review text"""
        print("\n[5/7] Cleaning review text...")

        def clean(text):
            """
            Removes whitespaces, non-English characters (Amharic, emojis, symbols).
            Keeps:
            - English letters
            - Numbers
            - Basic punctuation
            """
            if pd.isna(text) or text == "":
                return ""
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"[^A-Za-z0-9 .,!?;:'\"-]", " ", text)
            return str(text).strip()

        before = len(self.df)
        self.df["review_text"] = self.df["review_text"].apply(clean)
        self.df = self.df[self.df["review_text"].str.len() > 0]

        removed = before - len(self.df)
        if removed > 0:
            print(f"Removed {removed} empty/non-English reviews")

        self.stats["empty_reviews_removed"] = removed
        self.stats['count_after_cleaning'] = len(self.df)

        self.df["text_length"] = self.df["review_text"].str.len()

    # --------------------------------------------------------

    def validate_ratings(self):
        """Validate rating values (should be 1-5)"""
        print("\n[6/7] Validating rating values...")

        invalid = self.df[(self.df.rating < 1) | (self.df.rating > 5)]

        if len(invalid) > 0:
            print(f"WARNING: Found {len(invalid)} reviews with invalid ratings")
            self.df = self.df[(self.df.rating >= 1) & (self.df.rating <= 5)]
            print(f"Removed {len(invalid)} invalid ratings")
        else:
            print("All ratings are valid (1-5)")
        
        self.stats["invalid_ratings_removed"] = len(invalid)

    # --------------------------------------------------------

    def finalize(self):
        """Prepare final output format"""
        print("\n[7/7] Finalizing dataset...")

        output_columns = [
            "review_id",
            "review_text",
            "rating",
            "review_date",
            "review_year",
            "review_month",
            "bank_code",
            "bank_name",
            "user_name",
            "thumbs_up",
            "text_length",
            "source",
        ]

        # Filter the list to include only columns that actually exist in our DataFrame
        existing = [col for col in output_columns if col in self.df.columns]
        # Reorder the DataFrame columns
        self.df = self.df[existing]

        self.df = self.df.sort_values(["bank_code", "review_date"], ascending=[True, False])
        self.df = self.df.reset_index(drop=True)

        print(f"Final dataset: {len(self.df)} reviews")

    # --------------------------------------------------------

    def save_data(self):
        """Save processed data"""
        print("\nSaving processed data...")

        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.df.to_csv(self.output_path, index=False)
            print(f"\nSaved processed file to: {self.output_path}")

            self.stats['final_count'] = len(self.df)
            return True
        except Exception as e:
            print(f"ERROR: Failed to save data: {(e)}")
            return False

    # --------------------------------------------------------    

    def generate_report(self):
        """Generate preprocessing report"""
        print("\n" + "=" * 60)
        print("PREPROCESSING REPORT")
        print("=" * 60)

        # Print various statistics gathered during the process
        print(f"\nOriginal records: {self.stats.get('original_count', 0)}")
        print(f"Records with missing critical data: {self.stats.get('rows_removed_missing', 0)}")
        print(f"Duplicate reviews removed: {self.stats.get('duplicates_removed', 0)}")
        print(f"Empty reviews removed: {self.stats.get('empty_reviews_removed', 0)}")
        print(f"Invalid ratings removed: {self.stats.get('invalid_ratings_removed', 0)}")
        print(f"Final records: {self.stats.get('final_count', 0)}")

        # Calculate data quality percentage metrics
        if self.stats.get('original_count', 0) > 0:
            # Retention rate = (Final / Original) * 100
            retention_rate = (self.stats.get('final_count', 0) / self.stats.get('original_count', 1)) * 100
            # Error rate is the inverse of retention rate
            error_rate = 100 - retention_rate
            print(f"\nData retention rate: {retention_rate:.2f}%")
            print(f"Data error rate: {error_rate:.2f}%")

            # Assess quality based on error rate thresholds
            if error_rate < 5:
                print("✓ Data quality: EXCELLENT (<5% errors)")
            elif error_rate < 10:
                print("✓ Data quality: GOOD (<10% errors)")
            else:
                print("⚠ Data quality: NEEDS ATTENTION (>10% errors)")

        # Print statistics about the reviews per bank
        if self.df is not None:
            print("\nReviews per bank:")
            # Count occurrences of each unique value in 'bank_name'
            bank_counts = self.df['bank_name'].value_counts()
            for bank, count in bank_counts.items():
                print(f"  {bank}: {count}")

            # Print statistics about rating distribution
            print("\nRating distribution:")
            rating_counts = self.df['rating'].value_counts().sort_index(ascending=False)
            for rating, count in rating_counts.items():
                pct = (count / len(self.df)) * 100
                print(f"  {'⭐' * int(rating)}: {count} ({pct:.1f}%)")

            # Print the full date range of the data
            print(f"\nDate range: {self.df['review_date'].min()} to {self.df['review_date'].max()}")

            # Print statistics about the length of the review texts
            print(f"\nText statistics:")
            print(f"  Average length: {self.df['text_length'].mean():.0f} characters")
            print(f"  Median length: {self.df['text_length'].median():.0f} characters")
            print(f"  Min length: {self.df['text_length'].min()}")
            print(f"  Max length: {self.df['text_length'].max()}")

    # --------------------------------------------------------

    def process(self):
        """Run complete preprocessing pipeline"""
        print("=" * 60)
        print("STARTING PREPROCESSING")
        print("=" * 60)

        if not self.load_data():
            return False

        self.check_missing()
        self.remove_duplicates()
        self.handle_missing()
        self.normalize_dates()
        self.clean_text()
        self.validate_ratings()
        self.finalize()

        # Attempt to save the data. If successful, generate the report.
        if self.save_data():
            self.generate_report()
            return True

        # If saving failed, return False
        return False


def main():
    preprocessor = ReviewPreprocessor()
    success = preprocessor.process()

    if success:
        print("\n✅ Preprocessing completed successfully!")
        return preprocessor.df
    else:
        print("\n❌ Preprocessing failed!")
        return None


if __name__ == "__main__":
    processed_df = main()
