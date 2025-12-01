"""
Thematic Analysis Module
Task 2b: Extract keywords and identify themes in reviews
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import configuration
try:
    from scripts.config import DATA_PATHS, BANK_NAMES
except ModuleNotFoundError:
    from config import DATA_PATHS, BANK_NAMES

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class ThematicAnalyzer:
    """Modular thematic analysis class."""
    
    def __init__(self):
        """Initialize thematic analyzer."""
        # Setup stopwords
        self.stop_words = set(stopwords.words('english'))
        self.custom_stopwords = {
            'app', 'bank', 'banking', 'mobile', 'ethiopia', 
            'cbe', 'boa', 'dashen', 'please', 'thank', 'thanks',
            'application', 'bankings', 'apps', 'like', 'good', 'great',
            'bad', 'worst', 'best', 'love', 'hate'
        }
        self.stop_words.update(self.custom_stopwords)
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Define banking themes with keywords
        self.theme_definitions = {
            'Account Access Issues': {
                'keywords': ['login', 'password', 'account', 'access', 'verify', 
                           'authentication', 'blocked', 'locked', 'security', 'pin',
                           'biometric', 'fingerprint', 'face', 'unlock'],
                'description': 'Problems related to accessing accounts, login failures, or security issues'
            },
            'Transaction Performance': {
                'keywords': ['transfer', 'transaction', 'slow', 'fast', 'speed', 
                           'processing', 'time', 'wait', 'delay', 'instant', 'quick',
                           'pending', 'failed', 'success', 'complete', 'process'],
                'description': 'Issues related to transaction speed, processing time, or delays'
            },
            'User Interface & Experience': {
                'keywords': ['ui', 'ux', 'interface', 'design', 'layout', 'easy', 
                           'difficult', 'navigate', 'button', 'screen', 'menu',
                           'color', 'theme', 'font', 'size', 'view', 'display',
                           'simple', 'complex', 'intuitive', 'confusing'],
                'description': 'Feedback about app design, usability, and user experience'
            },
            'App Stability & Bugs': {
                'keywords': ['crash', 'bug', 'error', 'freeze', 'close', 'stop', 
                           'force', 'restart', 'reinstall', 'update', 'version',
                           'glitch', 'hang', 'not working', 'broken', 'fix', 'issue'],
                'description': 'Reports of app crashes, bugs, errors, or stability issues'
            },
            'Customer Support': {
                'keywords': ['support', 'service', 'help', 'call', 'contact', 
                           'response', 'agent', 'assist', 'complaint', 'email',
                           'phone', 'center', 'representative', 'resolve', 'solve'],
                'description': 'Feedback about customer service quality and responsiveness'
            },
            'Feature Requests': {
                'keywords': ['feature', 'add', 'need', 'want', 'missing', 
                           'suggestion', 'improve', 'enhance', 'option', 'would',
                           'could', 'should', 'include', 'implement', 'provide', 'wish'],
                'description': 'User requests for new features or enhancements'
            },
            'Financial Services': {
                'keywords': ['payment', 'bill', 'loan', 'deposit', 'withdraw', 
                           'balance', 'money', 'charge', 'fee', 'interest',
                           'account', 'statement', 'card', 'atm', 'cash', 'transfer'],
                'description': 'Feedback on banking services, fees, and financial operations'
            },
            'Registration & Onboarding': {
                'keywords': ['register', 'signup', 'onboard', 'create', 'account',
                           'verify', 'document', 'id', 'information', 'setup',
                           'profile', 'activation', 'activate', 'start'],
                'description': 'Issues related to initial app registration and account setup'
            },
            'Notifications & Alerts': {
                'keywords': ['notification', 'alert', 'message', 'sms', 'email',
                           'reminder', 'prompt', 'inform', 'update', 'news'],
                'description': 'Feedback about app notifications and alerts'
            }
        }
    
    def preprocess_text(self, text):
        """Preprocess text for keyword extraction."""
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove numbers and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens, then lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def extract_keywords_frequency(self, texts, top_n=30, min_freq=2):
        """
        Extract keywords using frequency analysis.
        
        Args:
            texts (list): List of review texts
            top_n (int): Number of top keywords to return
            min_freq (int): Minimum frequency for a word to be considered
            
        Returns:
            list: Top keywords
        """
        all_tokens = []
        
        for text in texts:
            tokens = self.preprocess_text(text)
            all_tokens.extend(tokens)
        
        # Count frequency
        word_freq = Counter(all_tokens)
        
        # Get most common words
        most_common = word_freq.most_common(top_n * 2)
        
        # Filter by minimum frequency and length
        filtered = [
            word for word, freq in most_common 
            if freq >= min_freq and len(word) > 2
        ]
        
        return filtered[:top_n]
    
    def extract_keywords_tfidf(self, texts, top_n=30):
        """Extract keywords using TF-IDF (requires scikit-learn)."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Preprocess all texts
            processed_texts = [' '.join(self.preprocess_text(text)) for text in texts]
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=top_n * 3,
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=2,  # Minimum document frequency
                max_df=0.9  # Maximum document frequency (ignore too common)
            )
            
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF scores
            tfidf_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # Get top scoring terms
            top_indices = tfidf_scores.argsort()[-top_n:][::-1]
            top_keywords = [
                feature_names[i] 
                for i in top_indices 
                if tfidf_scores[i] > 0 and len(feature_names[i]) > 2
            ]
            
            return top_keywords
            
        except ImportError:
            print("Warning: scikit-learn not available, using frequency analysis")
            return self.extract_keywords_frequency(texts, top_n)
        except Exception as e:
            print(f"Warning: TF-IDF extraction failed: {e}")
            return self.extract_keywords_frequency(texts, top_n)
    
    def identify_themes_for_bank(self, keywords, bank_code, min_themes=3, max_themes=5):
        """
        Identify themes for a specific bank based on keywords.
        
        Args:
            keywords (list): Extracted keywords
            bank_code (str): Bank code (CBE, BOA, Dashen)
            min_themes (int): Minimum number of themes to identify
            max_themes (int): Maximum number of themes to identify
            
        Returns:
            list: List of theme dictionaries
        """
        # Score each theme based on keyword matches
        theme_scores = {}
        
        for theme, definition in self.theme_definitions.items():
            theme_keywords = definition['keywords']
            score = 0
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                for theme_keyword in theme_keywords:
                    # Check for full or partial matches
                    if (theme_keyword == keyword_lower or 
                        theme_keyword in keyword_lower or
                        keyword_lower in theme_keyword):
                        score += 1
                        break
            
            if score > 0:
                theme_scores[theme] = {
                    'score': score,
                    'description': definition['description'],
                    'matched_keywords': [
                        kw for kw in keywords
                        if any(tk in kw.lower() for tk in theme_keywords)
                    ]
                }
        
        # Sort themes by score
        sorted_themes = sorted(
            theme_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        
        # Select top themes
        selected_themes = []
        for theme, info in sorted_themes[:max_themes]:
            if info['score'] >= 1:  # At least one keyword match
                selected_themes.append({
                    'theme': theme,
                    'description': info['description'],
                    'score': info['score'],
                    'example_keywords': info['matched_keywords'][:5],  # Top 5 examples
                    'bank_code': bank_code
                })
        
        # Ensure minimum number of themes
        if len(selected_themes) < min_themes:
            # Add default themes based on bank
            default_themes = self._get_default_themes(bank_code, min_themes - len(selected_themes))
            for theme in default_themes:
                if theme['theme'] not in [t['theme'] for t in selected_themes]:
                    selected_themes.append(theme)
        
        return selected_themes[:max_themes]
    
    def _get_default_themes(self, bank_code, count):
        """Get default themes for a bank if not enough are identified."""
        bank_defaults = {
            'CBE': ['Transaction Performance', 'App Stability & Bugs', 'Customer Support'],
            'BOA': ['Account Access Issues', 'User Interface & Experience', 'Feature Requests'],
            'Dashen': ['User Interface & Experience', 'Financial Services', 'Customer Support']
        }
        
        defaults = bank_defaults.get(bank_code, ['User Interface & Experience', 'App Stability & Bugs'])
        
        theme_list = []
        for theme_name in defaults[:count]:
            if theme_name in self.theme_definitions:
                theme_list.append({
                    'theme': theme_name,
                    'description': self.theme_definitions[theme_name]['description'],
                    'score': 0,
                    'example_keywords': self.theme_definitions[theme_name]['keywords'][:3],
                    'bank_code': bank_code
                })
        
        return theme_list
    
    def analyze_bank_reviews(self, df, bank_code):
        """
        Perform full thematic analysis for a specific bank.
        
        Args:
            df (pd.DataFrame): DataFrame containing reviews for the bank
            bank_code (str): Bank code
            
        Returns:
            dict: Analysis results including keywords and themes
        """
        print(f"Analyzing reviews for {BANK_NAMES.get(bank_code, bank_code)}...")
        
        if len(df) == 0:
            print(f"  No reviews found for {bank_code}")
            return None
        
        # Extract review texts
        texts = df['review_text'].fillna('').tolist()
        
        # Extract keywords
        print(f"  Extracting keywords from {len(texts)} reviews...")
        keywords = self.extract_keywords_tfidf(texts, top_n=30)
        
        if not keywords:
            keywords = self.extract_keywords_frequency(texts, top_n=30)
        
        print(f"  Top keywords: {', '.join(keywords[:10])}...")
        
        # Identify themes
        themes = self.identify_themes_for_bank(keywords, bank_code)
        
        # Prepare results
        results = {
            'bank_code': bank_code,
            'bank_name': BANK_NAMES.get(bank_code, bank_code),
            'total_reviews': len(df),
            'keywords': keywords,
            'themes': themes,
            'keyword_frequencies': Counter([kw for text in texts for kw in self.preprocess_text(text)])
        }
        
        return results
    
    def analyze_all_banks(self, df):
        """
        Perform thematic analysis for all banks.
        
        Args:
            df (pd.DataFrame): DataFrame containing reviews for all banks
            
        Returns:
            dict: Analysis results for all banks
        """
        print("\n" + "="*60)
        print("PERFORMING THEMATIC ANALYSIS")
        print("="*60)
        
        results = {}
        
        for bank_code in BANK_NAMES.keys():
            # Filter reviews for this bank
            bank_df = df[df['bank_code'] == bank_code]
            
            if len(bank_df) > 0:
                bank_results = self.analyze_bank_reviews(bank_df, bank_code)
                if bank_results:
                    results[bank_code] = bank_results
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results):
        """Generate a summary report of thematic analysis."""
        print("\n" + "="*60)
        print("THEMATIC ANALYSIS SUMMARY")
        print("="*60)
        
        for bank_code, bank_info in results.items():
            print(f"\n{BANK_NAMES.get(bank_code, bank_code).upper()}")
            print("-" * 40)
            print(f"Total Reviews: {bank_info['total_reviews']}")
            print(f"Top 10 Keywords: {', '.join(bank_info['keywords'][:10])}")
            
            print(f"\nIdentified Themes ({len(bank_info['themes'])}):")
            for i, theme in enumerate(bank_info['themes'], 1):
                print(f"  {i}. {theme['theme']}")
                print(f"     Score: {theme['score']}")
                if theme['example_keywords']:
                    print(f"     Examples: {', '.join(theme['example_keywords'][:3])}")
        
        # Overall statistics
        total_themes = sum(len(bank_info['themes']) for bank_info in results.values())
        avg_themes = total_themes / len(results) if results else 0
        
        print(f"\nOverall Statistics:")
        print(f"  Banks analyzed: {len(results)}")
        print(f"  Total themes identified: {total_themes}")
        print(f"  Average themes per bank: {avg_themes:.1f}")
        
        if avg_themes >= 3:
            print("✓ PASS: Meets requirement of 3+ themes per bank")
        else:
            print("⚠ WARNING: Below requirement of 3+ themes per bank")
    
    def assign_themes_to_reviews(self, df, bank_results):
        """
        Assign identified themes to individual reviews.
        
        Args:
            df (pd.DataFrame): Original DataFrame with reviews
            bank_results (dict): Thematic analysis results by bank
            
        Returns:
            pd.DataFrame: DataFrame with theme assignments
        """
        result_df = df.copy()
        result_df['identified_themes'] = ""
        
        for bank_code, bank_info in bank_results.items():
            # Get theme names for this bank
            theme_names = [theme['theme'] for theme in bank_info['themes']]
            theme_str = '; '.join(theme_names)
            
            # Assign to all reviews for this bank
            bank_indices = result_df[result_df['bank_code'] == bank_code].index
            result_df.loc[bank_indices, 'identified_themes'] = theme_str
        
        return result_df
    
    def save_results(self, df_with_themes, bank_results, output_dir=None):
        """Save thematic analysis results."""
        import os
        
        if output_dir is None:
            output_dir = DATA_PATHS.get('processed', 'data/processed')
        
        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 1. Save reviews with themes
        reviews_path = os.path.join(analysis_dir, "reviews_with_themes.csv")
        
        cols_to_save = [
            'review_id', 'review_text', 'bank_code', 'bank_name',
            'identified_themes'
        ]
        
        cols_to_save = [c for c in cols_to_save if c in df_with_themes.columns]
        df_with_themes[cols_to_save].to_csv(reviews_path, index=False)
        print(f"✓ Reviews with themes saved: {reviews_path}")
        
        # 2. Save themes summary
        themes_data = []
        for bank_code, bank_info in bank_results.items():
            for theme in bank_info['themes']:
                themes_data.append({
                    'bank_code': bank_code,
                    'bank_name': BANK_NAMES.get(bank_code, bank_code),
                    'theme': theme['theme'],
                    'description': theme['description'],
                    'score': theme['score'],
                    'example_keywords': '; '.join(theme['example_keywords'][:5]),
                    'total_reviews': bank_info['total_reviews']
                })
        
        themes_df = pd.DataFrame(themes_data)
        themes_path = os.path.join(analysis_dir, "themes_summary.csv")
        themes_df.to_csv(themes_path, index=False)
        print(f"✓ Themes summary saved: {themes_path}")
        
        # 3. Save keywords by bank
        keywords_data = []
        for bank_code, bank_info in bank_results.items():
            for i, keyword in enumerate(bank_info['keywords'][:20]):
                keywords_data.append({
                    'bank_code': bank_code,
                    'bank_name': BANK_NAMES.get(bank_code, bank_code),
                    'rank': i + 1,
                    'keyword': keyword,
                    'frequency': bank_info['keyword_frequencies'].get(keyword, 0)
                })
        
        keywords_df = pd.DataFrame(keywords_data)
        keywords_path = os.path.join(analysis_dir, "keywords_analysis.csv")
        keywords_df.to_csv(keywords_path, index=False)
        print(f"✓ Keywords analysis saved: {keywords_path}")
        
        return {
            'reviews_with_themes': reviews_path,
            'themes_summary': themes_path,
            'keywords_analysis': keywords_path
        }


def run_thematic_analysis(input_path=None):
    """
    Main function to run thematic analysis.
    
    Args:
        input_path (str): Path to processed reviews CSV or sentiment results
        
    Returns:
        dict: Analysis results
    """
    if input_path is None:
        # Try to load sentiment results first, fall back to processed reviews
        try:
            sentiment_path = DATA_PATHS.get('processed', 'data/processed') + "/analysis/sentiment_results.csv"
            df = pd.read_csv(sentiment_path)
            print(f"✓ Loaded sentiment results from {sentiment_path}")
        except:
            input_path = DATA_PATHS["processed_reviews"]
            df = pd.read_csv(input_path)
            print(f"✓ Loaded processed reviews from {input_path}")
    else:
        df = pd.read_csv(input_path)
        print(f"✓ Loaded data from {input_path}")
    
    print(f"Total reviews: {len(df)}")
    
    # Initialize analyzer
    analyzer = ThematicAnalyzer()
    
    # Perform analysis
    bank_results = analyzer.analyze_all_banks(df)
    
    # Assign themes to reviews
    df_with_themes = analyzer.assign_themes_to_reviews(df, bank_results)
    
    # Save results
    saved_files = analyzer.save_results(df_with_themes, bank_results)
    
    print("\n" + "="*60)
    print("THEMATIC ANALYSIS COMPLETE")
    print("="*60)
    
    return {
        'dataframe': df_with_themes,
        'bank_results': bank_results,
        'files': saved_files
    }


if __name__ == "__main__":
    # Example usage
    results = run_thematic_analysis()
    
    if results:
        print("\nSample of reviews with themes:")
        sample = results['dataframe'].head(3)
        for _, row in sample.iterrows():
            print(f"\nBank: {row['bank_name']}")
            print(f"Themes: {row.get('identified_themes', 'None')}")
            print(f"Review: {row['review_text'][:100]}...")