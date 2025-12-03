"""
PostgreSQL Database Implementation
Task 3: Store cleaned and processed review data in PostgreSQL
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import configuration
try:
    from scripts.config import DATA_PATHS, BANK_NAMES, APP_IDS, DATABASE_CONFIG
except ModuleNotFoundError:
    from config import DATA_PATHS, BANK_NAMES, APP_IDS, DATABASE_CONFIG


class BankReviewsDatabase:
    """Database manager for bank reviews analysis."""
    
    def __init__(self, config=None):
        """
        Initialize database connection.
        
        Args:
            config (dict): Database configuration dictionary
        """
        self.config = config or DATABASE_CONFIG
        self.connection = None
        self.cursor = None
        
        # Define bank app info (for banks table)
        self.bank_app_info = {
            'CBE': {
                'app_name': 'CBE Mobile Banking',
                'app_id': APP_IDS['CBE']
            },
            'BOA': {
                'app_name': 'BOA Mobile Banking', 
                'app_id': APP_IDS['BOA']
            },
            'Dashen': {
                'app_name': 'Dashen SuperApp',
                'app_id': APP_IDS['Dashen']
            }
        }
    
    def connect(self):
        """Establish connection to PostgreSQL database."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            print("Connecting to PostgreSQL database...")
            self.connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # Test connection
            self.cursor.execute("SELECT version();")
            db_version = self.cursor.fetchone()
            print(f"✓ Connected to PostgreSQL: {db_version['version']}")
            print(f"✓ Database: {self.config['database']}")
            
            return True
            
        except ImportError:
            print("❌ ERROR: psycopg2 not installed.")
            print("   Install with: pip install psycopg2-binary")
            return False
            
        except psycopg2.OperationalError as e:
            print(f"❌ ERROR: Could not connect to database: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure PostgreSQL is running: sudo service postgresql start")
            print("2. Check your credentials in .env file")
            print("3. Verify database exists: CREATE DATABASE bank_reviews;")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("\n✓ Database connection closed")
    
    def create_database(self):
        """Create the database if it doesn't exist."""
        try:
            # First connect without specifying database
            import psycopg2
            conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password']
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.config['database'],))
            exists = cursor.fetchone()
            
            if not exists:
                print(f"Creating database: {self.config['database']}")
                cursor.execute(f"CREATE DATABASE {self.config['database']}")
                print(f"✓ Database '{self.config['database']}' created")
            else:
                print(f"✓ Database '{self.config['database']}' already exists")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ ERROR creating database: {e}")
            return False
    
    def execute_sql_file(self, filepath):
        """Execute SQL commands from a file."""
        try:
            with open(filepath, 'r') as sql_file:
                sql_commands = sql_file.read()
            
            # Split by semicolon to execute commands separately
            commands = sql_commands.split(';')
            
            print(f"Executing SQL file: {filepath}")
            for command in commands:
                command = command.strip()
                if command and not command.startswith('--'):
                    try:
                        self.cursor.execute(command)
                    except Exception as e:
                        print(f"  ⚠ Warning executing command: {e}")
            
            self.connection.commit()
            print("✓ SQL file executed successfully")
            return True
            
        except FileNotFoundError:
            print(f"❌ SQL file not found: {filepath}")
            return False
        except Exception as e:
            print(f"❌ ERROR executing SQL file: {e}")
            return False
    
    def create_tables(self):
        """Create database tables using the schema file."""
        schema_file = os.path.join(os.path.dirname(__file__), 'db_schema.sql')
        
        if not os.path.exists(schema_file):
            print(f"❌ Schema file not found: {schema_file}")
            print("   Creating basic tables manually...")
            return self._create_basic_tables()
        
        return self.execute_sql_file(schema_file)
    
    def _create_basic_tables(self):
        """Create basic tables if schema file doesn't exist."""
        try:
            # Create banks table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS banks (
                    bank_id SERIAL PRIMARY KEY,
                    bank_code VARCHAR(10) UNIQUE NOT NULL,
                    bank_name VARCHAR(100) NOT NULL,
                    app_name VARCHAR(200),
                    app_id VARCHAR(100) UNIQUE,
                    current_rating DECIMAL(3,2),
                    total_reviews INTEGER DEFAULT 0,
                    total_ratings INTEGER DEFAULT 0,
                    installs VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create reviews table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id VARCHAR(100) PRIMARY KEY,
                    bank_id INTEGER NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
                    review_text TEXT NOT NULL,
                    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
                    sentiment_label VARCHAR(10) CHECK (sentiment_label IN ('positive', 'negative', 'neutral')),
                    sentiment_score DECIMAL(4,3) CHECK (sentiment_score BETWEEN 0 AND 1),
                    review_date DATE NOT NULL,
                    review_year INTEGER,
                    review_month INTEGER,
                    user_name VARCHAR(200),
                    thumbs_up INTEGER DEFAULT 0,
                    reply_content TEXT,
                    app_version VARCHAR(50),
                    source VARCHAR(50) DEFAULT 'Google Play',
                    text_length INTEGER,
                    has_speed_issue BOOLEAN DEFAULT FALSE,
                    has_feature_request BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.connection.commit()
            print("✓ Basic tables created successfully")
            return True
            
        except Exception as e:
            print(f"❌ ERROR creating tables: {e}")
            return False
    
    def insert_banks(self):
        """Insert bank information into banks table."""
        try:
            print("Inserting bank information...")
            
            for bank_code, bank_name in BANK_NAMES.items():
                # Check if bank already exists
                self.cursor.execute(
                    "SELECT bank_id FROM banks WHERE bank_code = %s",
                    (bank_code,)
                )
                exists = self.cursor.fetchone()
                
                if not exists:
                    # Insert new bank
                    self.cursor.execute("""
                        INSERT INTO banks (
                            bank_code, bank_name, app_name, app_id
                        ) VALUES (%s, %s, %s, %s)
                    """, (
                        bank_code,
                        bank_name,
                        self.bank_app_info[bank_code]['app_name'],
                        self.bank_app_info[bank_code]['app_id']
                    ))
                    print(f"  ✓ Inserted: {bank_name}")
            
            self.connection.commit()
            
            # Get bank IDs for reference
            self.cursor.execute("SELECT bank_code, bank_id FROM banks")
            self.bank_ids = {row['bank_code']: row['bank_id'] for row in self.cursor.fetchall()}
            
            print("✓ Bank information inserted successfully")
            return True
            
        except Exception as e:
            print(f"❌ ERROR inserting banks: {e}")
            return False
    
    def load_analysis_data(self):
        """Load the final analysis data from CSV."""
        try:
            data_path = DATA_PATHS['final_analysis']
            
            if not os.path.exists(data_path):
                print(f"⚠ Warning: Final analysis file not found: {data_path}")
                print("  Trying to load sentiment results instead...")
                data_path = DATA_PATHS['sentiment_results']
            
            if not os.path.exists(data_path):
                print(f"❌ ERROR: No analysis data found.")
                print(f"  Please run Task 2 analysis first.")
                return None
            
            df = pd.read_csv(data_path)
            print(f"✓ Loaded {len(df)} reviews from {data_path}")
            
            # Check required columns
            required_cols = ['review_id', 'review_text', 'rating', 'review_date', 'bank_code']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ ERROR: Missing required columns: {missing_cols}")
                return None
            
            # Clean and prepare data
            df = self._prepare_review_data(df)
            
            return df
            
        except Exception as e:
            print(f"❌ ERROR loading analysis data: {e}")
            return None
    
    def _prepare_review_data(self, df):
        """Prepare review data for database insertion."""
        # Make a copy to avoid modifying original
        df_prepared = df.copy()
        
        # Ensure review_id is string
        df_prepared['review_id'] = df_prepared['review_id'].astype(str)
        
        # Convert review_date to datetime
        if 'review_date' in df_prepared.columns:
            df_prepared['review_date'] = pd.to_datetime(df_prepared['review_date'], errors='coerce').dt.date
        
        # Extract year and month if not present
        if 'review_date' in df_prepared.columns and 'review_year' not in df_prepared.columns:
            df_prepared['review_year'] = pd.to_datetime(df_prepared['review_date']).dt.year
            df_prepared['review_month'] = pd.to_datetime(df_prepared['review_date']).dt.month
        
        # Calculate text length if not present
        if 'review_text' in df_prepared.columns and 'text_length' not in df_prepared.columns:
            df_prepared['text_length'] = df_prepared['review_text'].str.len()
        
        # Add speed issue flag
        if 'review_text' in df_prepared.columns and 'has_speed_issue' not in df_prepared.columns:
            speed_keywords = ['slow', 'fast', 'speed', 'loading', 'wait', 'delay', 'lag']
            df_prepared['has_speed_issue'] = df_prepared['review_text'].str.lower().apply(
                lambda x: any(keyword in str(x) for keyword in speed_keywords)
            )
        
        # Add feature request flag
        if 'review_text' in df_prepared.columns and 'has_feature_request' not in df_prepared.columns:
            feature_keywords = ['feature', 'add', 'need', 'want', 'missing', 'suggestion', 'improve']
            df_prepared['has_feature_request'] = df_prepared['review_text'].str.lower().apply(
                lambda x: any(keyword in str(x) for keyword in feature_keywords)
            )
        
        # Fill missing sentiment scores with neutral
        if 'sentiment_score' in df_prepared.columns:
            df_prepared['sentiment_score'] = df_prepared['sentiment_score'].fillna(0.5)
        
        # Fill missing sentiment labels
        if 'sentiment_label' in df_prepared.columns:
            df_prepared['sentiment_label'] = df_prepared['sentiment_label'].fillna('neutral')
        
        return df_prepared
    
    def insert_reviews(self, batch_size=100):
        """
        Insert reviews into the database.
        
        Args:
            batch_size (int): Number of reviews to insert at once
        """
        # Load data
        df = self.load_analysis_data()
        if df is None:
            return False
        
        print(f"\nInserting {len(df)} reviews into database...")
        
        # Get bank IDs
        if not hasattr(self, 'bank_ids'):
            self.cursor.execute("SELECT bank_code, bank_id FROM banks")
            self.bank_ids = {row['bank_code']: row['bank_id'] for row in self.cursor.fetchall()}
        
        # Prepare data for insertion
        reviews_inserted = 0
        reviews_skipped = 0
        reviews_updated = 0
        
        # Define the columns we want to insert
        review_columns = [
            'review_id', 'bank_id', 'review_text', 'rating',
            'sentiment_label', 'sentiment_score', 'review_date',
            'review_year', 'review_month', 'user_name', 'thumbs_up',
            'reply_content', 'app_version', 'source', 'text_length',
            'has_speed_issue', 'has_feature_request'
        ]
        
        # Process in batches
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                try:
                    # Map bank_code to bank_id
                    bank_code = row.get('bank_code')
                    if bank_code not in self.bank_ids:
                        print(f"  ⚠ Warning: Unknown bank code: {bank_code}")
                        reviews_skipped += 1
                        continue
                    
                    bank_id = self.bank_ids[bank_code]
                    
                    # Check if review already exists
                    self.cursor.execute(
                        "SELECT review_id FROM reviews WHERE review_id = %s",
                        (str(row.get('review_id', '')),)
                    )
                    exists = self.cursor.fetchone()
                    
                    if exists:
                        # Update existing review
                        self._update_review(row, bank_id)
                        reviews_updated += 1
                    else:
                        # Insert new review
                        self._insert_new_review(row, bank_id, review_columns)
                        reviews_inserted += 1
                        
                except Exception as e:
                    print(f"  ⚠ Error processing review: {e}")
                    reviews_skipped += 1
            
            # Commit after each batch
            self.connection.commit()
            
            # Show progress
            if (i + batch_size) % (batch_size * 5) == 0 or (i + batch_size) >= len(df):
                print(f"  Processed {min(i+batch_size, len(df))}/{len(df)} reviews...")
        
        # Final commit
        self.connection.commit()
        
        # Print summary
        print("\n" + "="*60)
        print("REVIEW INSERTION SUMMARY")
        print("="*60)
        print(f"Total reviews processed: {len(df)}")
        print(f"New reviews inserted: {reviews_inserted}")
        print(f"Existing reviews updated: {reviews_updated}")
        print(f"Reviews skipped: {reviews_skipped}")
        
        return reviews_inserted >= 400  # Check minimum requirement
    
    def _insert_new_review(self, row, bank_id, review_columns):
        """Insert a new review into the database."""
        # Prepare values dictionary
        values = {
            'review_id': str(row.get('review_id', '')),
            'bank_id': bank_id,
            'review_text': str(row.get('review_text', ''))[:10000],  # Limit text length
            'rating': int(row.get('rating', 3)),
            'sentiment_label': str(row.get('sentiment_label', 'neutral')).lower(),
            'sentiment_score': float(row.get('sentiment_score', 0.5)),
            'review_date': row.get('review_date'),
            'review_year': int(row.get('review_year', 2025)),
            'review_month': int(row.get('review_month', 12)),
            'user_name': str(row.get('user_name', 'Anonymous'))[:200],
            'thumbs_up': int(row.get('thumbs_up', 0)),
            'reply_content': str(row.get('reply_content', ''))[:5000],
            'app_version': str(row.get('app_version', 'N/A'))[:50],
            'source': str(row.get('source', 'Google Play'))[:50],
            'text_length': int(row.get('text_length', 0)),
            'has_speed_issue': bool(row.get('has_speed_issue', False)),
            'has_feature_request': bool(row.get('has_feature_request', False))
        }
        
        # Build INSERT query
        columns = ', '.join(values.keys())
        placeholders = ', '.join(['%s'] * len(values))
        
        query = f"""
            INSERT INTO reviews ({columns})
            VALUES ({placeholders})
        """
        
        self.cursor.execute(query, list(values.values()))
    
    def _update_review(self, row, bank_id):
        """Update an existing review in the database."""
        query = """
            UPDATE reviews 
            SET 
                review_text = %s,
                rating = %s,
                sentiment_label = %s,
                sentiment_score = %s,
                review_date = %s,
                review_year = %s,
                review_month = %s,
                user_name = %s,
                thumbs_up = %s,
                reply_content = %s,
                app_version = %s,
                source = %s,
                text_length = %s,
                has_speed_issue = %s,
                has_feature_request = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE review_id = %s
        """
        
        values = (
            str(row.get('review_text', ''))[:10000],
            int(row.get('rating', 3)),
            str(row.get('sentiment_label', 'neutral')).lower(),
            float(row.get('sentiment_score', 0.5)),
            row.get('review_date'),
            int(row.get('review_year', 2025)),
            int(row.get('review_month', 12)),
            str(row.get('user_name', 'Anonymous'))[:200],
            int(row.get('thumbs_up', 0)),
            str(row.get('reply_content', ''))[:5000],
            str(row.get('app_version', 'N/A'))[:50],
            str(row.get('source', 'Google Play'))[:50],
            int(row.get('text_length', 0)),
            bool(row.get('has_speed_issue', False)),
            bool(row.get('has_feature_request', False)),
            str(row.get('review_id', ''))
        )
        
        self.cursor.execute(query, values)
    
    def verify_data_integrity(self):
        """Run SQL queries to verify data integrity."""
        print("\n" + "="*60)
        print("VERIFYING DATA INTEGRITY")
        print("="*60)
        
        verification_passed = True
        
        try:
            # Query 1: Count reviews per bank
            print("\n1. Reviews per bank:")
            self.cursor.execute("""
                SELECT 
                    b.bank_name,
                    COUNT(r.review_id) as review_count
                FROM banks b
                LEFT JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_id, b.bank_name
                ORDER BY review_count DESC
            """)
            
            results = self.cursor.fetchall()
            for row in results:
                print(f"  {row['bank_name']}: {row['review_count']} reviews")
                if row['review_count'] < 400:
                    print(f"    ⚠ Warning: Below minimum of 400 reviews")
                    verification_passed = False
            
            # Query 2: Average rating per bank
            print("\n2. Average rating per bank:")
            self.cursor.execute("""
                SELECT 
                    b.bank_name,
                    ROUND(AVG(r.rating)::numeric, 2) as avg_rating,
                    COUNT(r.review_id) as count
                FROM banks b
                LEFT JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_id, b.bank_name
                ORDER BY avg_rating DESC
            """)
            
            results = self.cursor.fetchall()
            for row in results:
                print(f"  {row['bank_name']}: {row['avg_rating']} (from {row['count']} reviews)")
            
            # Query 3: Sentiment distribution
            print("\n3. Sentiment distribution:")
            self.cursor.execute("""
                SELECT 
                    sentiment_label,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
                FROM reviews
                GROUP BY sentiment_label
                ORDER BY count DESC
            """)
            
            results = self.cursor.fetchall()
            for row in results:
                print(f"  {row['sentiment_label']}: {row['count']} ({row['percentage']}%)")
            
            # Query 4: Date range
            print("\n4. Date range of reviews:")
            self.cursor.execute("""
                SELECT 
                    MIN(review_date) as earliest,
                    MAX(review_date) as latest,
                    COUNT(*) as total_reviews
                FROM reviews
            """)
            
            result = self.cursor.fetchone()
            print(f"  From: {result['earliest']}")
            print(f"  To: {result['latest']}")
            print(f"  Total reviews in database: {result['total_reviews']}")
            
            # Query 5: Check for NULL values in critical columns
            print("\n5. Data quality check:")
            critical_columns = ['review_text', 'rating', 'review_date']
            
            for column in critical_columns:
                self.cursor.execute(f"""
                    SELECT COUNT(*) as null_count
                    FROM reviews
                    WHERE {column} IS NULL
                """)
                result = self.cursor.fetchone()
                if result['null_count'] > 0:
                    print(f"  ⚠ {column}: {result['null_count']} NULL values")
                    verification_passed = False
                else:
                    print(f"  ✓ {column}: No NULL values")
            
            # Query 6: Rating distribution
            print("\n6. Rating distribution:")
            self.cursor.execute("""
                SELECT 
                    rating,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
                FROM reviews
                GROUP BY rating
                ORDER BY rating DESC
            """)
            
            results = self.cursor.fetchall()
            for row in results:
                stars = '⭐' * row['rating']
                print(f"  {stars}: {row['count']} ({row['percentage']}%)")
            
            return verification_passed
            
        except Exception as e:
            print(f"❌ ERROR during verification: {e}")
            return False
    
    def generate_schema_documentation(self):
        """Generate schema documentation in README format."""
        try:
            print("\n" + "="*60)
            print("GENERATING SCHEMA DOCUMENTATION")
            print("="*60)
            
            # Get table information
            self.cursor.execute("""
                SELECT 
                    table_name,
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
            """)
            
            tables = {}
            for row in self.cursor.fetchall():
                table_name = row['table_name']
                if table_name not in tables:
                    tables[table_name] = []
                tables[table_name].append(row)
            
            # Create documentation
            docs_dir = DATA_PATHS['database']
            docs_file = os.path.join(docs_dir, 'database_schema.md')
            
            with open(docs_file, 'w') as f:
                f.write("# Database Schema Documentation\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for table_name, columns in tables.items():
                    f.write(f"## Table: `{table_name}`\n\n")
                    
                    # Create markdown table
                    f.write("| Column | Type | Nullable | Default | Description |\n")
                    f.write("|--------|------|----------|---------|-------------|\n")
                    
                    for col in columns:
                        # Add description based on column name
                        description = self._get_column_description(col['column_name'], table_name)
                        f.write(f"| {col['column_name']} | {col['data_type']} | {col['is_nullable']} | {col['column_default'] or ''} | {description} |\n")
                    
                    f.write("\n")
            
            print(f"✓ Schema documentation saved to: {docs_file}")
            
            # Also save SQL schema
            schema_file = os.path.join(docs_dir, 'schema_dump.sql')
            self.cursor.execute("""
                SELECT 
                    table_name,
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
            """)
            
            with open(schema_file, 'w') as f:
                f.write("-- Database Schema Dump\n")
                f.write(f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for table_name, columns in tables.items():
                    f.write(f"-- Table: {table_name}\n")
                    f.write(f"CREATE TABLE {table_name} (\n")
                    
                    col_defs = []
                    for col in columns:
                        col_def = f"    {col['column_name']} {col['data_type'].upper()}"
                        if col['is_nullable'] == 'NO':
                            col_def += " NOT NULL"
                        if col['column_default']:
                            col_def += f" DEFAULT {col['column_default']}"
                        col_defs.append(col_def)
                    
                    f.write(',\n'.join(col_defs))
                    f.write("\n);\n\n")
            
            print(f"✓ SQL schema dump saved to: {schema_file}")
            
            return True
            
        except Exception as e:
            print(f"⚠ Warning: Could not generate documentation: {e}")
            return False
    
    def _get_column_description(self, column_name, table_name):
        """Get description for a column based on its name and table."""
        descriptions = {
            'banks': {
                'bank_id': 'Primary key, auto-incrementing',
                'bank_code': 'Unique code for the bank (CBE, BOA, Dashen)',
                'bank_name': 'Full name of the bank',
                'app_name': 'Name of the mobile banking app',
                'app_id': 'Google Play Store app ID',
                'current_rating': 'Current app rating on Play Store',
                'total_reviews': 'Total number of reviews on Play Store',
                'total_ratings': 'Total number of ratings on Play Store',
                'installs': 'Number of app installs',
                'created_at': 'Timestamp when record was created',
                'updated_at': 'Timestamp when record was last updated'
            },
            'reviews': {
                'review_id': 'Primary key, unique review ID from Play Store',
                'bank_id': 'Foreign key referencing banks table',
                'review_text': 'Text content of the review',
                'rating': 'Star rating (1-5)',
                'sentiment_label': 'Sentiment classification (positive/negative/neutral)',
                'sentiment_score': 'Sentiment confidence score (0-1)',
                'review_date': 'Date when review was posted',
                'review_year': 'Year extracted from review_date',
                'review_month': 'Month extracted from review_date',
                'user_name': 'Name of the user who posted review',
                'thumbs_up': 'Number of helpful votes',
                'reply_content': 'Developer reply to the review',
                'app_version': 'App version when review was posted',
                'source': 'Source of review (Google Play)',
                'text_length': 'Length of review text in characters',
                'has_speed_issue': 'Flag indicating if review mentions speed issues',
                'has_feature_request': 'Flag indicating if review contains feature requests',
                'created_at': 'Timestamp when record was inserted',
                'updated_at': 'Timestamp when record was last updated'
            }
        }
        
        if table_name in descriptions and column_name in descriptions[table_name]:
            return descriptions[table_name][column_name]
        return 'No description available'
    
    def run_database_pipeline(self):
        """Run complete database setup and data insertion pipeline."""
        print("="*70)
        print("STARTING TASK 3: DATABASE IMPLEMENTATION")
        print("="*70)
        
        # Step 1: Create database if it doesn't exist
        print("\n📦 STEP 1: Database Setup")
        print("-" * 40)
        if not self.create_database():
            print("⚠ Continuing with existing database...")
        
        # Step 2: Connect to database
        if not self.connect():
            return False
        
        try:
            # Step 3: Create tables
            print("\n📋 STEP 2: Creating Tables")
            print("-" * 40)
            if not self.create_tables():
                print("⚠ Table creation had issues, but continuing...")
            
            # Step 4: Insert bank information
            print("\n🏦 STEP 3: Inserting Bank Information")
            print("-" * 40)
            if not self.insert_banks():
                print("⚠ Bank insertion had issues, but continuing...")
            
            # Step 5: Insert reviews
            print("\n📝 STEP 4: Inserting Reviews")
            print("-" * 40)
            if not self.insert_reviews(batch_size=100):
                print("⚠ Review insertion had issues, but continuing...")
            
            # Step 6: Verify data integrity
            print("\n✅ STEP 5: Verifying Data Integrity")
            print("-" * 40)
            verification_passed = self.verify_data_integrity()
            
            # Step 7: Generate documentation
            print("\n📄 STEP 6: Generating Documentation")
            print("-" * 40)
            self.generate_schema_documentation()
            
            # Step 8: Summary
            print("\n" + "="*70)
            print("TASK 3 COMPLETION SUMMARY")
            print("="*70)
            
            if verification_passed:
                print("✅ DATABASE IMPLEMENTATION SUCCESSFUL!")
            else:
                print("⚠ DATABASE IMPLEMENTATION COMPLETED WITH WARNINGS")
            
            print("\nNext Steps:")
            print("1. Check the database with: psql -U postgres -d bank_reviews")
            print("2. Run sample queries from the schema documentation")
            print("3. Proceed to Task 4: Insights and Recommendations")
            
            return verification_passed
            
        finally:
            # Always disconnect
            self.disconnect()


# =============================================
# QUERY EXAMPLES FOR VERIFICATION
# =============================================

class DatabaseQueries:
    """Example queries for data verification and analysis."""
    
    @staticmethod
    def get_sample_queries():
        """Return sample SQL queries for verification."""
        return {
            "count_reviews_per_bank": """
                SELECT 
                    b.bank_name,
                    COUNT(r.review_id) as total_reviews,
                    ROUND(AVG(r.rating)::numeric, 2) as average_rating
                FROM banks b
                LEFT JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_id, b.bank_name
                ORDER BY total_reviews DESC;
            """,
            
            "sentiment_analysis_summary": """
                SELECT 
                    b.bank_name,
                    r.sentiment_label,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY b.bank_name), 2) as percentage
                FROM banks b
                JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_name, r.sentiment_label
                ORDER BY b.bank_name, count DESC;
            """,
            
            "monthly_review_trends": """
                SELECT 
                    b.bank_name,
                    r.review_year,
                    r.review_month,
                    COUNT(*) as review_count,
                    ROUND(AVG(r.rating)::numeric, 2) as avg_rating
                FROM banks b
                JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_name, r.review_year, r.review_month
                ORDER BY b.bank_name, r.review_year DESC, r.review_month DESC;
            """,
            
            "top_rated_banks": """
                SELECT 
                    b.bank_name,
                    ROUND(AVG(r.rating)::numeric, 2) as avg_rating,
                    COUNT(r.review_id) as review_count
                FROM banks b
                JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_id, b.bank_name
                ORDER BY avg_rating DESC
                LIMIT 3;
            """,
            
            "speed_issues_analysis": """
                SELECT 
                    b.bank_name,
                    SUM(CASE WHEN r.has_speed_issue THEN 1 ELSE 0 END) as speed_issues,
                    COUNT(*) as total_reviews,
                    ROUND(SUM(CASE WHEN r.has_speed_issue THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as percentage
                FROM banks b
                JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_name
                ORDER BY percentage DESC;
            """
        }


# =============================================
# MAIN EXECUTION
# =============================================

def main():
    """Main function to run the database pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bank Reviews Database Implementation')
    parser.add_argument('--create-only', action='store_true', help='Only create database and tables')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing data')
    parser.add_argument('--skip-insert', action='store_true', help='Skip data insertion')
    
    args = parser.parse_args()
    
    # Initialize database manager
    db = BankReviewsDatabase()
    
    if args.verify_only:
        # Only verify existing data
        if db.connect():
            db.verify_data_integrity()
            db.disconnect()
        return
    
    if args.create_only:
        # Only create database and tables
        db.create_database()
        if db.connect():
            db.create_tables()
            db.insert_banks()
            db.disconnect()
        return
    
    # Run full pipeline
    success = db.run_database_pipeline()
    
    if success:
        print("\n✅ Task 3 completed successfully!")
        print("\nTo test the database, run these commands:")
        print("  psql -U postgres -d bank_reviews -c 'SELECT * FROM bank_summary;'")
        print("  psql -U postgres -d bank_reviews -c 'SELECT COUNT(*) FROM reviews;'")
    else:
        print("\n❌ Task 3 encountered issues!")
        print("Check the error messages above and troubleshoot.")


if __name__ == "__main__":
    main()