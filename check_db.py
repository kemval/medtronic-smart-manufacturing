import sqlite3
from pathlib import Path

def check_database():
    db_path = Path('data/manufacturing.db')
    print(f"Checking database at: {db_path.absolute()}")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in the database!")
            return
            
        print("\nFound tables:")
        for table in tables:
            table_name = table[0]
            print(f"\nTable: {table_name}")
            
            # Get column info
            try:
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                print("Columns:")
                for col in columns:
                    print(f"  - {col[1]} ({col[2]})")
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                count = cursor.fetchone()[0]
                print(f"  Rows: {count}")
                
            except sqlite3.Error as e:
                print(f"  Error reading table info: {e}")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_database()
