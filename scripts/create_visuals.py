"""
Simplified runner for Task 4
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.insights import main as insights_main
    from scripts.report_generator import main as report_main
except ImportError:
    from insights import main as insights_main
    from report_generator import main as report_main

def run_full_task4():
    """Run complete Task 4 pipeline."""
    print("="*70)
    print("RUNNING TASK 4: INSIGHTS AND RECOMMENDATIONS")
    print("="*70)
    
    print("\n📊 STEP 1: Generating Insights...")
    print("-" * 40)
    
    # Import and run insights
    from scripts.insights import InsightsGenerator
    insights_gen = InsightsGenerator()
    insights_success = insights_gen.run_insights_pipeline()
    
    if not insights_success:
        print("❌ Insights generation failed!")
        return False
    
    print("\n📄 STEP 2: Generating Final Report...")
    print("-" * 40)
    
    # Import and run report generator
    from scripts.report_generator import ReportGenerator
    report_gen = ReportGenerator()
    report_success = report_gen.run_report_generation('final')
    
    if not report_success:
        print("❌ Report generation failed!")
        return False
    
    print("\n" + "="*70)
    print("TASK 4 COMPLETE!")
    print("="*70)
    
    print("\n📋 Deliverables Generated:")
    print("1. Insights summary (JSON/CSV)")
    print("2. Recommendations matrix")
    print("3. 5+ Visualizations")
    print("5. Scenario analysis results")
    
    print("\n📍 Files located in:")
    print(f"   - Insights: {os.path.join('data', 'processed', 'insights')}")
    print(f"   - Reports: {os.path.join('reports')}")
    print(f"   - Visualizations: {os.path.join('reports', 'visualizations')}")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Task 4 pipeline')
    parser.add_argument('--insights-only', action='store_true', help='Only generate insights')
    parser.add_argument('--report-only', action='store_true', help='Only generate report')
    parser.add_argument('--interim', action='store_true', help='Generate interim report')
    
    args = parser.parse_args()
    
    if args.insights_only:
        insights_main()
    elif args.report_only:
        report_main()
    elif args.interim:
        from scripts.report_generator import ReportGenerator
        rg = ReportGenerator()
        rg.run_report_generation('interim')
    else:
        run_full_task4()