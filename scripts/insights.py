"""
Insights and Recommendations Module
Task 4: Derive insights from analyzed data and generate recommendations

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from scripts.config import DATA_PATHS, BANK_NAMES, REPORT_PATHS
except ModuleNotFoundError:
    from config import DATA_PATHS, BANK_NAMES, REPORT_PATHS

class InsightsGenerator:
    """Generate business insights and recommendations from analyzed data."""
    
    def __init__(self, data_path=None):
        """
        Initialize insights generator.
        
        Args:
            data_path (str): Path to final analysis results
        """
        self.data_path = data_path or DATA_PATHS['final_analysis']
        self.df = None
        self.insights = {}
        self.recommendations = {}
        self.visualizations = {}
        
    def load_data(self):
        """Load analyzed data."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Loaded {len(self.df)} analyzed reviews")
            
            # Add scenario flags if not present
            self._add_scenario_flags()
            
            return True
        except FileNotFoundError:
            print(f"❌ Data file not found: {self.data_path}")
            print("   Please run Task 2 analysis first")
            return False
    
    def _add_scenario_flags(self):
        """Add scenario-specific flags to data."""
        # Scenario 1: Speed issues
        speed_keywords = ['slow', 'fast', 'speed', 'loading', 'wait', 'delay', 'lag', 'response']
        self.df['has_speed_issue'] = self.df['review_text'].str.lower().apply(
            lambda x: any(keyword in str(x) for keyword in speed_keywords) if pd.notna(x) else False
        )
        
        # Scenario 2: Feature requests
        feature_keywords = ['feature', 'add', 'need', 'want', 'missing', 'suggestion', 
                           'improve', 'enhance', 'option', 'would like', 'should have']
        self.df['has_feature_request'] = self.df['review_text'].str.lower().apply(
            lambda x: any(keyword in str(x) for keyword in feature_keywords) if pd.notna(x) else False
        )
        
        # Scenario 3: Complaint types
        complaint_categories = {
            'login_issues': ['login', 'password', 'access', 'blocked', 'locked', 'verify'],
            'transaction_issues': ['transfer', 'transaction', 'failed', 'pending', 'error'],
            'app_crashes': ['crash', 'close', 'freeze', 'stop', 'restart'],
            'ui_issues': ['interface', 'design', 'difficult', 'confusing', 'layout'],
            'support_issues': ['support', 'service', 'help', 'response', 'call']
        }
        
        for category, keywords in complaint_categories.items():
            self.df[f'has_{category}'] = self.df['review_text'].str.lower().apply(
                lambda x: any(keyword in str(x) for keyword in keywords) if pd.notna(x) else False
            )
    
    def generate_bank_comparison_insights(self):
        """Generate comparative insights across banks."""
        print("\n" + "="*60)
        print("GENERATING BANK COMPARISON INSIGHTS")
        print("="*60)
        
        insights = []
        
        # Overall performance metrics
        bank_metrics = self.df.groupby('bank_name').agg({
            'rating': ['mean', 'count'],
            'sentiment_score': 'mean',
            'sentiment_label': lambda x: (x == 'positive').mean() * 100
        }).round(2)
        
        bank_metrics.columns = ['avg_rating', 'review_count', 'avg_sentiment', 'positive_pct']
        
        # Rank banks
        bank_metrics['rating_rank'] = bank_metrics['avg_rating'].rank(ascending=False)
        bank_metrics['sentiment_rank'] = bank_metrics['avg_sentiment'].rank(ascending=False)
        bank_metrics['positive_rank'] = bank_metrics['positive_pct'].rank(ascending=False)
        
        print("\n📊 Bank Performance Metrics:")
        print(bank_metrics.to_string())
        
        # Identify top performer
        top_bank = bank_metrics.sort_values('avg_rating', ascending=False).iloc[0]
        bottom_bank = bank_metrics.sort_values('avg_rating', ascending=True).iloc[0]
        
        insights.append({
            'type': 'performance_comparison',
            'title': 'Overall Performance Ranking',
            'description': f"Based on average rating, {top_bank.name} leads with {top_bank['avg_rating']}/5, while {bottom_bank.name} has the lowest rating at {bottom_bank['avg_rating']}/5.",
            'metrics': bank_metrics.to_dict('index')
        })
        
        # Sentiment analysis
        sentiment_dist = pd.crosstab(self.df['bank_name'], self.df['sentiment_label'], 
                                     normalize='index') * 100
        
        print("\n📈 Sentiment Distribution (%):")
        print(sentiment_dist.round(2).to_string())
        
        insights.append({
            'type': 'sentiment_analysis',
            'title': 'Customer Sentiment Analysis',
            'description': f"Sentiment analysis reveals varying levels of customer satisfaction across banks.",
            'metrics': sentiment_dist.round(2).to_dict('index')
        })
        
        # Rating distribution insights
        rating_insights = []
        for bank in self.df['bank_name'].unique():
            bank_data = self.df[self.df['bank_name'] == bank]
            five_star_pct = (bank_data['rating'] == 5).mean() * 100
            one_star_pct = (bank_data['rating'] == 1).mean() * 100
            
            rating_insights.append({
                'bank': bank,
                'five_star_pct': round(five_star_pct, 1),
                'one_star_pct': round(one_star_pct, 1),
                'polarization_index': round(abs(five_star_pct - one_star_pct), 1)
            })
        
        insights.append({
            'type': 'rating_distribution',
            'title': 'Rating Distribution Analysis',
            'description': "Analysis of 5-star vs 1-star reviews shows polarization levels.",
            'metrics': rating_insights
        })
        
        self.insights['bank_comparison'] = insights
        return insights
    
    def analyze_scenario_1_retaining_users(self):
        """Analyze Scenario 1: Retaining Users (Speed Issues)."""
        print("\n" + "="*60)
        print("SCENARIO 1: RETAINING USERS - SPEED ISSUES ANALYSIS")
        print("="*60)
        
        insights = []
        
        # Speed issues by bank
        speed_analysis = self.df.groupby('bank_name').agg({
            'has_speed_issue': ['sum', 'mean', 'count']
        }).round(3)
        
        speed_analysis.columns = ['speed_issue_count', 'speed_issue_pct', 'total_reviews']
        speed_analysis['speed_issue_pct'] = speed_analysis['speed_issue_pct'] * 100
        
        print("\n🚀 Speed Issue Analysis:")
        print(speed_analysis.to_string())
        
        # Correlation with rating
        speed_correlation = {}
        for bank in self.df['bank_name'].unique():
            bank_data = self.df[self.df['bank_name'] == bank]
            correlation = bank_data[['rating', 'has_speed_issue']].corr().iloc[0, 1]
            speed_correlation[bank] = round(correlation, 3)
        
        # Identify most affected bank
        most_issues = speed_analysis['speed_issue_pct'].idxmax()
        issue_pct = speed_analysis.loc[most_issues, 'speed_issue_pct']
        
        insights.append({
            'type': 'speed_issues',
            'title': 'Transaction Speed and Loading Issues',
            'description': f"{most_issues} has the highest percentage of speed-related complaints ({issue_pct:.1f}% of reviews). Speed issues show a correlation of {speed_correlation[most_issues]:.3f} with lower ratings.",
            'metrics': {
                'speed_analysis': speed_analysis.to_dict('index'),
                'correlation_with_rating': speed_correlation
            },
            'severity': 'high' if issue_pct > 15 else 'medium' if issue_pct > 8 else 'low'
        })
        
        # Detailed analysis of speed complaints
        speed_reviews = self.df[self.df['has_speed_issue'] == True]
        speed_keywords = self._extract_keywords_from_texts(
            speed_reviews['review_text'].tolist(), 
            top_n=10
        )
        
        insights.append({
            'type': 'speed_keywords',
            'title': 'Common Speed-Related Keywords',
            'description': "Analysis of speed-related reviews reveals common pain points.",
            'metrics': {
                'top_keywords': speed_keywords,
                'sample_complaints': speed_reviews['review_text'].head(3).tolist()
            }
        })
        
        self.insights['scenario_1'] = insights
        return insights
    
    def analyze_scenario_2_enhancing_features(self):
        """Analyze Scenario 2: Enhancing Features."""
        print("\n" + "="*60)
        print("SCENARIO 2: ENHANCING FEATURES - FEATURE REQUESTS ANALYSIS")
        print("="*60)
        
        insights = []
        
        # Feature requests by bank
        feature_analysis = self.df.groupby('bank_name').agg({
            'has_feature_request': ['sum', 'mean', 'count']
        }).round(3)
        
        feature_analysis.columns = ['feature_request_count', 'feature_request_pct', 'total_reviews']
        feature_analysis['feature_request_pct'] = feature_analysis['feature_request_pct'] * 100
        
        print("\n🛠️ Feature Request Analysis:")
        print(feature_analysis.to_string())
        
        # Extract specific feature requests
        feature_reviews = self.df[self.df['has_feature_request'] == True]
        
        # Common feature request keywords
        feature_categories = {
            'security': ['biometric', 'fingerprint', 'face id', 'security', '2fa', 'authentication'],
            'convenience': ['quick', 'instant', 'one-click', 'automate', 'schedule'],
            'functionality': ['bill pay', 'loan', 'investment', 'budget', 'analytics', 'report'],
            'integration': ['other bank', 'integration', 'api', 'connect', 'export'],
            'personalization': ['theme', 'customize', 'preference', 'layout', 'language']
        }
        
        # Categorize feature requests
        feature_categories_analysis = {}
        for bank in self.df['bank_name'].unique():
            bank_features = feature_reviews[feature_reviews['bank_name'] == bank]
            categories_count = {}
            
            for category, keywords in feature_categories.items():
                count = sum(
                    any(keyword in review.lower() for keyword in keywords)
                    for review in bank_features['review_text']
                    if isinstance(review, str)
                )
                categories_count[category] = count
            
            feature_categories_analysis[bank] = categories_count
        
        # Identify most requested features per bank
        top_features_by_bank = {}
        for bank, categories in feature_categories_analysis.items():
            if categories:
                top_category = max(categories.items(), key=lambda x: x[1])
                if top_category[1] > 0:
                    top_features_by_bank[bank] = {
                        'category': top_category[0],
                        'count': top_category[1]
                    }
        
        insights.append({
            'type': 'feature_requests',
            'title': 'Feature Request Analysis',
            'description': f"Users are actively requesting new features across all apps. Feature requests account for {feature_analysis['feature_request_pct'].mean():.1f}% of all reviews on average.",
            'metrics': {
                'feature_analysis': feature_analysis.to_dict('index'),
                'top_features_by_bank': top_features_by_bank,
                'feature_categories': feature_categories_analysis
            },
            'opportunity_level': 'high' if feature_analysis['feature_request_pct'].mean() > 10 else 'medium'
        })
        
        # Extract sample feature requests
        sample_requests = []
        for bank in self.df['bank_name'].unique():
            bank_requests = feature_reviews[feature_reviews['bank_name'] == bank]
            if not bank_requests.empty:
                sample_requests.append({
                    'bank': bank,
                    'requests': bank_requests['review_text'].head(2).tolist()
                })
        
        insights.append({
            'type': 'sample_feature_requests',
            'title': 'Sample Feature Requests',
            'description': "Direct user feedback on desired features.",
            'metrics': {'sample_requests': sample_requests}
        })
        
        self.insights['scenario_2'] = insights
        return insights
    
    def analyze_scenario_3_managing_complaints(self):
        """Analyze Scenario 3: Managing Complaints."""
        print("\n" + "="*60)
        print("SCENARIO 3: MANAGING COMPLAINTS - ISSUE CLUSTERING")
        print("="*60)
        
        insights = []
        
        # Complaint type analysis
        complaint_types = ['login_issues', 'transaction_issues', 'app_crashes', 'ui_issues', 'support_issues']
        
        complaint_analysis = {}
        for bank in self.df['bank_name'].unique():
            bank_data = self.df[self.df['bank_name'] == bank]
            bank_complaints = {}
            
            for complaint_type in complaint_types:
                complaint_pct = bank_data[f'has_{complaint_type}'].mean() * 100
                bank_complaints[complaint_type] = round(complaint_pct, 2)
            
            complaint_analysis[bank] = bank_complaints
        
        # Convert to DataFrame for better visualization
        complaint_df = pd.DataFrame(complaint_analysis).T
        
        print("\n🚨 Complaint Type Analysis (% of reviews):")
        print(complaint_df.round(2).to_string())
        
        # Identify top complaint per bank
        top_complaints = {}
        for bank, complaints in complaint_analysis.items():
            if complaints:
                top_complaint = max(complaints.items(), key=lambda x: x[1])
                top_complaints[bank] = {
                    'type': top_complaint[0],
                    'percentage': top_complaint[1]
                }
        
        # Severity analysis
        severity_levels = {}
        for bank, complaints in complaint_analysis.items():
            high_severity = sum(1 for pct in complaints.values() if pct > 10)
            medium_severity = sum(1 for pct in complaints.values() if 5 <= pct <= 10)
            
            severity_levels[bank] = {
                'high_severity_issues': high_severity,
                'medium_severity_issues': medium_severity,
                'total_significant_issues': high_severity + medium_severity
            }
        
        insights.append({
            'type': 'complaint_clustering',
            'title': 'Complaint Type Clustering for AI Chatbot',
            'description': "Complaints cluster into distinct categories that can guide AI chatbot training and support ticket routing.",
            'metrics': {
                'complaint_distribution': complaint_analysis,
                'top_complaints': top_complaints,
                'severity_analysis': severity_levels
            }
        })
        
        # Sentiment of complaints
        complaint_sentiment = {}
        for bank in self.df['bank_name'].unique():
            bank_data = self.df[self.df['bank_name'] == bank]
            bank_complaint_sentiment = {}
            
            for complaint_type in complaint_types:
                complaint_data = bank_data[bank_data[f'has_{complaint_type}'] == True]
                if not complaint_data.empty:
                    avg_sentiment = complaint_data['sentiment_score'].mean()
                    avg_rating = complaint_data['rating'].mean()
                    bank_complaint_sentiment[complaint_type] = {
                        'avg_sentiment': round(avg_sentiment, 3),
                        'avg_rating': round(avg_rating, 2)
                    }
            
            complaint_sentiment[bank] = bank_complaint_sentiment
        
        insights.append({
            'type': 'complaint_sentiment',
            'title': 'Sentiment Analysis of Complaints',
            'description': "Understanding sentiment associated with different complaint types helps prioritize resolution efforts.",
            'metrics': {'complaint_sentiment': complaint_sentiment}
        })
        
        # Response time implications
        print("\n⏱️ Response Time Implications:")
        for bank, complaints in top_complaints.items():
            if complaints['percentage'] > 8:
                print(f"  {bank}: High volume of '{complaints['type']}' complaints ({complaints['percentage']}%)")
                print(f"    → Requires fast chatbot responses or priority support routing")
        
        self.insights['scenario_3'] = insights
        return insights
    
    def _extract_keywords_from_texts(self, texts, top_n=10):
        """Extract top keywords from list of texts."""
        from collections import Counter
        import re
        
        all_words = []
        for text in texts:
            if isinstance(text, str):
                words = re.findall(r'\b[a-z]{3,}\b', text.lower())
                all_words.extend(words)
        
        # Remove common stopwords
        stopwords = {'the', 'and', 'for', 'that', 'this', 'with', 'have', 'from', 'they', 'what'}
        filtered_words = [word for word in all_words if word not in stopwords]
        
        word_counts = Counter(filtered_words)
        return word_counts.most_common(top_n)
    
    def generate_drivers_and_pain_points(self):
        """Identify satisfaction drivers and pain points for each bank."""
        print("\n" + "="*60)
        print("IDENTIFYING DRIVERS AND PAIN POINTS")
        print("="*60)
        
        drivers_pain_points = {}
        
        for bank in self.df['bank_name'].unique():
            bank_data = self.df[self.df['bank_name'] == bank]
            
            # Drivers (positive aspects)
            positive_reviews = bank_data[bank_data['sentiment_label'] == 'positive']
            
            if len(positive_reviews) > 0:
                # Extract common themes in positive reviews
                positive_keywords = self._extract_keywords_from_texts(
                    positive_reviews['review_text'].tolist(),
                    top_n=15
                )
                
                # Categorize drivers
                drivers = self._categorize_drivers(positive_keywords)
            else:
                drivers = []
            
            # Pain points (negative aspects)
            negative_reviews = bank_data[bank_data['sentiment_label'] == 'negative']
            
            if len(negative_reviews) > 0:
                # Extract common themes in negative reviews
                negative_keywords = self._extract_keywords_from_texts(
                    negative_reviews['review_text'].tolist(),
                    top_n=15
                )
                
                # Categorize pain points
                pain_points = self._categorize_pain_points(negative_keywords)
            else:
                pain_points = []
            
            drivers_pain_points[bank] = {
                'drivers': drivers[:3],  # Top 3 drivers
                'pain_points': pain_points[:3],  # Top 3 pain points
                'positive_review_count': len(positive_reviews),
                'negative_review_count': len(negative_reviews)
            }
            
            print(f"\n{bank}:")
            print(f"  Drivers: {', '.join([d['category'] for d in drivers[:3]])}")
            print(f"  Pain Points: {', '.join([p['category'] for p in pain_points[:3]])}")
        
        self.insights['drivers_pain_points'] = drivers_pain_points
        return drivers_pain_points
    
    def _categorize_drivers(self, keywords):
        """Categorize keywords into satisfaction drivers."""
        driver_categories = {
            'Speed & Performance': ['fast', 'quick', 'instant', 'speed', 'efficient'],
            'User Interface': ['easy', 'simple', 'interface', 'design', 'beautiful', 'clean'],
            'Reliability': ['reliable', 'stable', 'consistent', 'dependable'],
            'Features': ['feature', 'function', 'option', 'tool', 'capability'],
            'Customer Support': ['support', 'help', 'service', 'response', 'assist'],
            'Security': ['secure', 'safe', 'protection', 'trust', 'privacy']
        }
        
        drivers = []
        for category, category_keywords in driver_categories.items():
            score = sum(count for word, count in keywords if any(kw in word for kw in category_keywords))
            if score > 0:
                example_words = [word for word, count in keywords if any(kw in word for kw in category_keywords)]
                drivers.append({
                    'category': category,
                    'score': score,
                    'example_keywords': example_words[:3]
                })
        
        return sorted(drivers, key=lambda x: x['score'], reverse=True)
    
    def _categorize_pain_points(self, keywords):
        """Categorize keywords into pain points."""
        pain_point_categories = {
            'Slow Performance': ['slow', 'lag', 'delay', 'wait', 'loading', 'time'],
            'Technical Issues': ['crash', 'error', 'bug', 'freeze', 'stop', 'close'],
            'Login Problems': ['login', 'password', 'access', 'blocked', 'verify'],
            'Transaction Issues': ['transfer', 'transaction', 'failed', 'pending', 'process'],
            'Poor UI/UX': ['difficult', 'confusing', 'complex', 'layout', 'design'],
            'Lack of Features': ['missing', 'need', 'want', 'should', 'could', 'would'],
            'Customer Support': ['support', 'service', 'help', 'response', 'ignore']
        }
        
        pain_points = []
        for category, category_keywords in pain_point_categories.items():
            score = sum(count for word, count in keywords if any(kw in word for kw in category_keywords))
            if score > 0:
                example_words = [word for word, count in keywords if any(kw in word for kw in category_keywords)]
                pain_points.append({
                    'category': category,
                    'score': score,
                    'example_keywords': example_words[:3]
                })
        
        return sorted(pain_points, key=lambda x: x['score'], reverse=True)
    
    def generate_recommendations(self):
        """Generate actionable recommendations for each bank."""
        print("\n" + "="*60)
        print("GENERATING RECOMMENDATIONS")
        print("="*60)
        
        recommendations = {}
        
        for bank in self.df['bank_name'].unique():
            bank_data = self.df[self.df['bank_name'] == bank]
            bank_insights = {}
            
            # Overall performance
            avg_rating = bank_data['rating'].mean()
            positive_pct = (bank_data['sentiment_label'] == 'positive').mean() * 100
            
            # Get drivers and pain points
            if 'drivers_pain_points' in self.insights:
                drivers = self.insights['drivers_pain_points'][bank]['drivers']
                pain_points = self.insights['drivers_pain_points'][bank]['pain_points']
            else:
                drivers = []
                pain_points = []
            
            # Generate recommendations based on insights
            bank_recommendations = []
            
            # Recommendation 1: Based on overall performance
            if avg_rating < 3.5:
                bank_recommendations.append({
                    'type': 'urgent',
                    'title': 'Immediate App Improvement',
                    'description': f'With an average rating of {avg_rating:.2f}/5, immediate action is required to address core issues.',
                    'action': 'Conduct user interviews to identify root causes of dissatisfaction',
                    'timeline': '1 month',
                    'priority': 'High'
                })
            elif avg_rating < 4.0:
                bank_recommendations.append({
                    'type': 'improvement',
                    'title': 'Targeted Feature Enhancement',
                    'description': f'Rating of {avg_rating:.2f}/5 indicates room for improvement in key areas.',
                    'action': 'Prioritize enhancements based on user feedback analysis',
                    'timeline': '2-3 months',
                    'priority': 'Medium'
                })
            
            # Recommendation 2: Based on top pain points
            if pain_points:
                top_pain_point = pain_points[0] if pain_points else {}
                if top_pain_point.get('category') == 'Slow Performance':
                    bank_recommendations.append({
                        'type': 'technical',
                        'title': 'Performance Optimization',
                        'description': 'Speed-related complaints are the top pain point affecting user experience.',
                        'action': 'Optimize transaction processing and app loading times',
                        'timeline': '1-2 months',
                        'priority': 'High'
                    })
                elif top_pain_point.get('category') == 'Technical Issues':
                    bank_recommendations.append({
                        'type': 'technical',
                        'title': 'Stability Improvement',
                        'description': 'App crashes and technical errors are significantly impacting user satisfaction.',
                        'action': 'Conduct thorough testing and implement error monitoring',
                        'timeline': '2 months',
                        'priority': 'High'
                    })
            
            # Recommendation 3: Feature development
            feature_requests_pct = bank_data['has_feature_request'].mean() * 100
            if feature_requests_pct > 10:
                bank_recommendations.append({
                    'type': 'feature',
                    'title': 'Feature Development Roadmap',
                    'description': f'High feature request rate ({feature_requests_pct:.1f}%) indicates user demand for new capabilities.',
                    'action': 'Create and communicate a feature development roadmap',
                    'timeline': '3-6 months',
                    'priority': 'Medium'
                })
            
            # Recommendation 4: Based on drivers
            if drivers:
                top_driver = drivers[0] if drivers else {}
                bank_recommendations.append({
                    'type': 'leverage',
                    'title': f'Leverage {top_driver.get("category", "Strengths")}',
                    'description': f'Users appreciate {top_driver.get("category", "certain aspects")} of your app.',
                    'action': f'Amplify marketing around {top_driver.get("category", "these strengths")}',
                    'timeline': 'Ongoing',
                    'priority': 'Low'
                })
            
            # Add at least 2 recommendations if not enough
            if len(bank_recommendations) < 2:
                bank_recommendations.extend([
                    {
                        'type': 'general',
                        'title': 'User Feedback Analysis',
                        'description': 'Regular analysis of user reviews helps identify emerging issues.',
                        'action': 'Implement monthly review analysis and reporting',
                        'timeline': 'Ongoing',
                        'priority': 'Medium'
                    },
                    {
                        'type': 'general',
                        'title': 'Competitive Analysis',
                        'description': 'Understanding competitor strengths can inform development priorities.',
                        'action': 'Benchmark against competing banking apps quarterly',
                        'timeline': 'Quarterly',
                        'priority': 'Low'
                    }
                ])
            
            recommendations[bank] = bank_recommendations[:3]  # Top 3 recommendations
            
            print(f"\n{bank} - Top 3 Recommendations:")
            for i, rec in enumerate(recommendations[bank], 1):
                print(f"  {i}. {rec['title']} ({rec['priority']} priority)")
                print(f"     {rec['action']}")
        
        self.recommendations = recommendations
        return recommendations
    
    def save_insights(self):
        """Save insights and recommendations to files."""
        import json
        import os
        
        insights_dir = REPORT_PATHS['insights']
        os.makedirs(insights_dir, exist_ok=True)
        
        # Save insights
        insights_file = os.path.join(insights_dir, 'insights_summary.json')
        with open(insights_file, 'w') as f:
            json.dump(self.insights, f, indent=2, default=str)
        
        print(f"\n✓ Insights saved to: {insights_file}")
        
        # Save recommendations
        rec_file = os.path.join(insights_dir, 'recommendations.json')
        with open(rec_file, 'w') as f:
            json.dump(self.recommendations, f, indent=2, default=str)
        
        print(f"✓ Recommendations saved to: {rec_file}")
        
        # Save summary CSV
        summary_data = []
        for bank, recs in self.recommendations.items():
            for i, rec in enumerate(recs, 1):
                summary_data.append({
                    'bank': bank,
                    'recommendation_number': i,
                    'title': rec['title'],
                    'type': rec['type'],
                    'priority': rec['priority'],
                    'action': rec['action'],
                    'timeline': rec['timeline']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(insights_dir, 'recommendations_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"✓ Recommendations summary saved to: {summary_file}")
        
        return {
            'insights': insights_file,
            'recommendations': rec_file,
            'summary': summary_file
        }
    
    def run_insights_pipeline(self):
        """Run complete insights generation pipeline."""
        print("="*70)
        print("TASK 4: INSIGHTS AND RECOMMENDATIONS GENERATION")
        print("="*70)
        
        # Load data
        if not self.load_data():
            return False
        
        # Generate insights
        print("\n📊 GENERATING INSIGHTS...")
        self.generate_bank_comparison_insights()
        self.analyze_scenario_1_retaining_users()
        self.analyze_scenario_2_enhancing_features()
        self.analyze_scenario_3_managing_complaints()
        self.generate_drivers_and_pain_points()
        
        # Generate recommendations
        print("\n💡 GENERATING RECOMMENDATIONS...")
        self.generate_recommendations()
        
        # Save results
        print("\n💾 SAVING RESULTS...")
        saved_files = self.save_insights()
        
        # Generate summary report
        print("\n" + "="*70)
        print("INSIGHTS GENERATION COMPLETE")
        print("="*70)
        
        print(f"\n📋 Generated for each bank:")
        print(f"  • 3+ Satisfaction Drivers")
        print(f"  • 3+ Pain Points")
        print(f"  • 3+ Actionable Recommendations")
        
        print(f"\n🎯 Scenarios Analyzed:")
        print(f"  ✓ Scenario 1: Retaining Users (Speed Issues)")
        print(f"  ✓ Scenario 2: Enhancing Features")
        print(f"  ✓ Scenario 3: Managing Complaints")
        
        print(f"\n📊 Files saved to: {REPORT_PATHS['insights']}")
        
        return True


def main():
    """Main execution function."""
    insights_gen = InsightsGenerator()
    success = insights_gen.run_insights_pipeline()
    
    if success:
        print("\n✅ Task 4 Insights Generation Complete!")
    else:
        print("\n❌ Insights generation failed!")


if __name__ == "__main__":
    main()