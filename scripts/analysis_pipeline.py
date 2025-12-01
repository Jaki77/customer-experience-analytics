"""
Analysis Pipeline Orchestrator
Task 2: Orchestrates sentiment and thematic analysis
"""

import argparse
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.config import DATA_PATHS
    from scripts.sentiment import run_sentiment_analysis
    from scripts.themes import run_thematic_analysis
except ImportError:
    from config import DATA_PATHS
    from sentiment import run_sentiment_analysis
    from themes import run_thematic_analysis


def run_full_analysis(method='distilbert', input_path=None):
    """
    Run complete Task 2 analysis pipeline.
    
    Args:
        method (str): Sentiment analysis method
        input_path (str): Path to input data
        
    Returns:
        bool: True if successful
    """
    print("="*70)
    print("STARTING TASK 2 ANALYSIS PIPELINE")
    print("="*70)
    
    # Step 1: Sentiment Analysis
    print("\n📊 PHASE 1: SENTIMENT ANALYSIS")
    print("-" * 40)
    
    sentiment_results = run_sentiment_analysis(
        input_path=input_path,
        method=method
    )
    
    if sentiment_results is None:
        print("❌ Sentiment analysis failed. Exiting.")
        return False
    
    # Step 2: Thematic Analysis
    print("\n🎯 PHASE 2: THEMATIC ANALYSIS")
    print("-" * 40)
    
    thematic_results = run_thematic_analysis()
    
    if thematic_results is None:
        print("❌ Thematic analysis failed. Exiting.")
        return False
    
    # Step 3: Merge Results
    print("\n🔄 PHASE 3: MERGING RESULTS")
    print("-" * 40)
    
    try:
        # Load both results
        sentiment_path = DATA_PATHS.get('processed', 'data/processed') + "/analysis/sentiment_results.csv"
        themes_path = DATA_PATHS.get('processed', 'data/processed') + "/analysis/reviews_with_themes.csv"
        
        import pandas as pd
        sentiment_df = pd.read_csv(sentiment_path)
        themes_df = pd.read_csv(themes_path)
        
        # Merge on review_id
        if 'review_id' in sentiment_df.columns and 'review_id' in themes_df.columns:
            merged_df = pd.merge(
                sentiment_df,
                themes_df[['review_id', 'identified_themes']],
                on='review_id',
                how='left'
            )
            
            # Save merged results
            output_path = DATA_PATHS.get('processed', 'data/processed') + "/analysis/final_analysis_results.csv"
            merged_df.to_csv(output_path, index=False)
            print(f"✓ Merged results saved to: {output_path}")
            print(f"✓ Total records: {len(merged_df)}")
            
            # Final report
            print("\n📋 FINAL REPORT")
            print("-" * 40)
            
            banks = merged_df['bank_name'].unique()
            print(f"Banks analyzed: {', '.join(banks)}")
            
            total_reviews = len(merged_df)
            analyzed_reviews = merged_df['sentiment_label'].notna().sum()
            coverage = (analyzed_reviews / total_reviews) * 100
            
            print(f"\nCoverage Statistics:")
            print(f"  Total reviews: {total_reviews}")
            print(f"  Reviews with sentiment: {analyzed_reviews} ({coverage:.1f}%)")
            
            if coverage >= 90:
                print("  ✓ PASS: Meets 90% coverage requirement")
            else:
                print("  ⚠ WARNING: Below 90% coverage requirement")
            
            # Themes per bank
            print(f"\nThemes per bank:")
            for bank in banks:
                bank_themes = thematic_results['bank_results'].get(
                    merged_df[merged_df['bank_name'] == bank]['bank_code'].iloc[0], {}
                )
                theme_count = len(bank_themes.get('themes', []))
                print(f"  {bank}: {theme_count} themes")
                
                if theme_count >= 3:
                    print(f"    ✓ PASS: Meets 3+ themes requirement")
                else:
                    print(f"    ⚠ WARNING: Below 3 themes requirement")
            
            print("\n✅ TASK 2 COMPLETE!")
            print("Next: Run visualization notebook")
            
            return True
            
        else:
            print("⚠ WARNING: Could not merge - review_id column missing")
            return True
            
    except Exception as e:
        print(f"⚠ WARNING: Could not merge results: {e}")
        return True


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run Task 2 analysis pipeline for bank reviews"
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='distilbert',
        choices=['distilbert', 'vader', 'textblob'],
        help='Sentiment analysis method'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input CSV file (default: data/processed/reviews_processed.csv)'
    )
    
    parser.add_argument(
        '--skip-sentiment',
        action='store_true',
        help='Skip sentiment analysis'
    )
    
    parser.add_argument(
        '--skip-themes',
        action='store_true',
        help='Skip thematic analysis'
    )
    
    args = parser.parse_args()
    
    if args.skip_sentiment and args.skip_themes:
        print("Error: Cannot skip both sentiment and thematic analysis")
        return
    
    # Run appropriate analysis
    if not args.skip_sentiment:
        run_sentiment_analysis(input_path=args.input, method=args.method)
    
    if not args.skip_themes:
        run_thematic_analysis()
    
    # Only run full pipeline if neither is skipped
    if not (args.skip_sentiment or args.skip_themes):
        run_full_analysis(method=args.method, input_path=args.input)


if __name__ == "__main__":
    main()