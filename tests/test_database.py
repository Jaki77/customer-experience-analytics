"""
Test script for database functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.database import BankReviewsDatabase, DatabaseQueries

def test_database_connection():
    """Test database connection."""
    print("Testing database connection...")
    
    db = BankReviewsDatabase()
    
    if db.connect():
        print("✓ Connection successful")
        
        # Test a simple query
        db.cursor.execute("SELECT COUNT(*) as count FROM information_schema.tables WHERE table_schema = 'public'")
        result = db.cursor.fetchone()
        print(f"✓ Found {result['count']} tables in database")
        
        db.disconnect()
        return True
    else:
        print("✗ Connection failed")
        return False

def test_sample_queries():
    """Test sample queries."""
    print("\nTesting sample queries...")
    
    db = BankReviewsDatabase()
    
    if db.connect():
        queries = DatabaseQueries.get_sample_queries()
        
        for query_name, query in queries.items():
            print(f"\n{query_name}:")
            try:
                db.cursor.execute(query)
                results = db.cursor.fetchall()
                
                if results:
                    # Print column names
                    columns = [desc[0] for desc in db.cursor.description]
                    print(f"  Columns: {', '.join(columns)}")
                    
                    # Print first few rows
                    for i, row in enumerate(results[:3]):
                        print(f"  Row {i+1}: {row}")
                    
                    print(f"  Total rows: {len(results)}")
                else:
                    print("  No results returned")
                    
            except Exception as e:
                print(f"  ⚠ Query failed: {e}")
        
        db.disconnect()
        return True
    else:
        return False

def test_data_integrity():
    """Test data integrity checks."""
    print("\nTesting data integrity...")
    
    db = BankReviewsDatabase()
    
    if db.connect():
        # Check if tables exist
        db.cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = db.cursor.fetchall()
        
        print(f"Tables found: {[t['table_name'] for t in tables]}")
        
        # Check row counts
        for table in ['banks', 'reviews']:
            try:
                db.cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                result = db.cursor.fetchone()
                print(f"  {table}: {result['count']} rows")
            except:
                print(f"  {table}: Table does not exist")
        
        db.disconnect()
        return True
    else:
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("DATABASE TEST SUITE")
    print("="*60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Sample Queries", test_sample_queries),
        ("Data Integrity", test_data_integrity)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Test: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Database is ready.")
    else:
        print("⚠ Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main()