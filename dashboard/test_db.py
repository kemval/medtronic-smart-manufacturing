import os
import sys
import sqlite3
from pathlib import Path

def test_database_connection():
    print("Testing database connection...")
    
    # Set up database path
    db_dir = Path('data')
    db_dir.mkdir(exist_ok=True, parents=True)
    db_path = db_dir / 'manufacturing.db'
    
    print(f"Database path: {db_path.absolute()}")
    
    try:
        # Test connection
        conn = sqlite3.connect(str(db_path))
        print("✅ Successfully connected to database")
        
        # Check tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if tables:
            print("\nFound tables:")
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"- {table_name}: {count} rows")
        else:
            print("\nNo tables found in the database")
            
        # Create test table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_connection (
            id INTEGER PRIMARY KEY,
            test_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # Insert test data
        cursor.execute("INSERT INTO test_connection (test_text) VALUES (?)", 
                      ("Test connection successful",))
        conn.commit()
        
        print("\n✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()
    
    return True

if __name__ == "__main__":
    test_database_connection()
