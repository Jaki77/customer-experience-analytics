"""
Sentiment Analysis Module
Task 2a: Analyze review sentiment using DistilBERT with fallbacks
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from scripts.config import DATA_PATHS
except ModuleNotFoundError:
    from config import DATA_PATHS

class SentimentAnalyzer:
    """Modular sentiment analysis class."""
    
    def __init__(self, method='distilbert'):
        """
        Initialize sentiment analyzer.
        
        Args:
            method (str): 'distilbert', 'vader', or 'textblob'
        """
        self.method = self._validate_method(method)
        self.analyzer = self._initialize_analyzer()
        
    def _validate_method(self, method):
        """Validate and set sentiment analysis method."""
        valid_methods = ['distilbert', 'vader', 'textblob']
        if method not in valid_methods:
            print(f"Warning: {method} not valid. Using 'distilbert'")
            return 'distilbert'
        return method
    
    def _initialize_analyzer(self):
        """Initialize the chosen sentiment analyzer."""
        print(f"Initializing {self.method.upper()} sentiment analyzer...")
        
        if self.method == 'distilbert':
            try:
                from transformers import pipeline
                analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    truncation=True,
                    max_length=512
                )
                print("✓ DistilBERT loaded successfully")
                return analyzer
            except ImportError:
                print("Warning: Transformers not available, falling back to TextBlob")
                self.method = 'textblob'
                
        if self.method == 'vader':
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()
                print("✓ VADER loaded successfully")
                return analyzer
            except ImportError:
                print("Warning: VADER not available, falling back to TextBlob")
                self.method = 'textblob'
                
        if self.method == 'textblob':
            try:
                from textblob import TextBlob
                print("✓ TextBlob loaded successfully")
                return None  # TextBlob doesn't need initialization
            except ImportError:
                print("ERROR: No sentiment analysis library available!")
                raise
    
    def analyze_text_distilbert(self, text):
        """Analyze sentiment using DistilBERT."""
        try:
            if len(text) > 4000:
                text = text[:4000]
            
            result = self.analyzer(text[:512])[0]
            label = result['label'].upper()
            score = result['score']
            
            if label in ['POSITIVE', 'LABEL_1']:
                return 'positive', score
            elif label in ['NEGATIVE', 'LABEL_0']:
                return 'negative', score
            else:
                return 'neutral', 0.5
                
        except Exception as e:
            print(f"Warning: DistilBERT error: {e}")
            return 'neutral', 0.5
    
    def analyze_text_vader(self, text):
        """Analyze sentiment using VADER."""
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'positive', compound
        elif compound <= -0.05:
            return 'negative', abs(compound)
        else:
            return 'neutral', 0.5
    
    def analyze_text_textblob(self, text):
        """Analyze sentiment using TextBlob."""
        try:
            from textblob import TextBlob
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            
            if polarity > 0.1:
                return 'positive', polarity
            elif polarity < -0.1:
                return 'negative', abs(polarity)
            else:
                return 'neutral', 0.5
        except Exception:
            return 'neutral', 0.5
    
    def analyze_single(self, text):
        """Analyze sentiment of a single text."""
        if not isinstance(text, str) or len(text.strip()) < 3:
            return 'neutral', 0.5
        
        if self.method == 'distilbert':
            return self.analyze_text_distilbert(text)
        elif self.method == 'vader':
            return self.analyze_text_vader(text)
        else:  # textblob
            return self.analyze_text_textblob(text)
    
    def analyze_batch(self, texts, batch_size=50, verbose=True):
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (list): List of text strings
            batch_size (int): Number of texts to process at once
            verbose (bool): Show progress
            
        Returns:
            tuple: (sentiments, scores)
        """
        sentiments = []
        scores = []
        
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            
            for text in batch:
                sentiment, score = self.analyze_single(text)
                sentiments.append(sentiment)
                scores.append(score)
            
            if verbose and (i // batch_size) % 5 == 0:
                print(f"Processed {min(i+batch_size, total)}/{total} texts...")
        
        if verbose:
            print(f"✓ Completed sentiment analysis for {total} texts")
        
        return sentiments, scores
    
    def analyze_dataframe(self, df, text_column='review_text'):
        """
        Add sentiment analysis to a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing reviews
            text_column (str): Name of column with review text
            
        Returns:
            pd.DataFrame: DataFrame with added sentiment columns
        """
        print(f"\nStarting sentiment analysis on {len(df)} reviews...")
        print(f"Method: {self.method.upper()}")
        
        # Get texts
        texts = df[text_column].fillna('').tolist()
        
        # Analyze
        sentiments, scores = self.analyze_batch(texts, verbose=True)
        
        # Add to DataFrame
        result_df = df.copy()
        result_df['sentiment_label'] = sentiments
        result_df['sentiment_score'] = scores
        
        # Report statistics
        self._report_statistics(result_df)
        
        return result_df
    
    def _report_statistics(self, df):
        """Generate sentiment analysis statistics."""
        print("\n" + "="*50)
        print("SENTIMENT ANALYSIS RESULTS")
        print("="*50)
        
        total = len(df)
        sentiment_counts = df['sentiment_label'].value_counts()
        
        print(f"Total reviews analyzed: {total}")
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total) * 100
            print(f"  {sentiment.upper():8s}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\nAverage sentiment score: {df['sentiment_score'].mean():.3f}")
        print(f"Sentiment score std dev: {df['sentiment_score'].std():.3f}")
        
        # By rating
        print("\nSentiment by Rating:")
        rating_summary = df.groupby('rating').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).round(3)
        
        print(rating_summary.to_string())
        
        # Coverage
        analyzed = df['sentiment_label'].notna().sum()
        coverage = (analyzed / total) * 100
        print(f"\nCoverage: {coverage:.1f}% ({analyzed}/{total} reviews)")
        
        if coverage >= 90:
            print("✓ PASS: Coverage meets 90% minimum requirement")
        else:
            print("⚠ WARNING: Coverage below 90% minimum requirement")
    
    def save_results(self, df, output_path=None):
        """Save sentiment analysis results."""
        if output_path is None:
            output_dir = DATA_PATHS.get('processed', 'data/processed')
            import os
            os.makedirs(f"{output_dir}/analysis", exist_ok=True)
            output_path = f"{output_dir}/analysis/sentiment_results.csv"
        
        # Save only relevant columns
        cols_to_save = [
            'review_id', 'review_text', 'rating', 'review_date',
            'bank_code', 'bank_name', 'sentiment_label', 'sentiment_score'
        ]
        
        cols_to_save = [c for c in cols_to_save if c in df.columns]
        
        df[cols_to_save].to_csv(output_path, index=False)
        print(f"\n✓ Sentiment results saved to: {output_path}")
        
        return output_path


def run_sentiment_analysis(input_path=None, method='distilbert'):
    """
    Main function to run sentiment analysis.
    
    Args:
        input_path (str): Path to processed reviews CSV
        method (str): Sentiment analysis method
        
    Returns:
        pd.DataFrame: DataFrame with sentiment analysis results
    """
    if input_path is None:
        input_path = DATA_PATHS["processed_reviews"]
    
    print("="*60)
    print("RUNNING SENTIMENT ANALYSIS")
    print("="*60)
    
    # Load data
    try:
        df = pd.read_csv(input_path)
        print(f"✓ Loaded {len(df)} reviews from {input_path}")
    except FileNotFoundError:
        print(f"ERROR: File not found: {input_path}")
        print("Please run preprocessing first (python preprocess.py)")
        return None
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(method=method)
    
    # Perform analysis
    result_df = analyzer.analyze_dataframe(df)
    
    # Save results
    output_file = analyzer.save_results(result_df)
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS COMPLETE")
    print("="*60)
    
    return result_df


if __name__ == "__main__":
    # Example usage
    df_result = run_sentiment_analysis()
    
    if df_result is not None:
        print("\nSample of results:")
        print(df_result[['review_text', 'rating', 'sentiment_label', 'sentiment_score']].head(5))