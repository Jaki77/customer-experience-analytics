"""
Simplified runner for Task 3
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.database import main as db_main
except ImportError:
    from database import main as db_main

if __name__ == "__main__":
    print("="*60)
    print("RUNNING TASK 3: DATABASE IMPLEMENTATION")
    print("="*60)
    
    # Run the database pipeline
    db_main()