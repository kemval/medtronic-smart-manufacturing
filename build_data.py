"""
Master Data Pipeline
Runs all data generation and processing steps
Save as: build_data.py (in project root)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_processing.process_cnc_data import CNCDataProcessor
from data_generation.generate_synthetic_data import ManufacturingDataGenerator
import pandas as pd
import sqlite3

def create_database():
    """Create SQLite database with all data"""
    print("\n" + "="*60)
    print("🗄️  CREATING SQLITE DATABASE")
    print("="*60 + "\n")
    
    db_path = Path('data/manufacturing.db')
    
    # Remove existing database
    if db_path.exists():
        db_path.unlink()
        print("🗑️  Removed existing database")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    print(f"✅ Created database: {db_path}")
    
    # Load all processed data
    print("\n📂 Loading processed data files...")
    data_dir = Path('data/processed')
    
    tables = {
        'production': data_dir / 'production_data.csv',
        'quality': data_dir / 'quality_data.csv',
        'maintenance': data_dir / 'maintenance_data.csv',
        'cnc_features': data_dir / 'cnc_features.csv',
        'cnc_sensors': data_dir / 'cnc_sensor_data.csv'
    }
    
    # Load and save to database
    for table_name, filepath in tables.items():
        if not filepath.exists():
            print(f"⚠️  Warning: {filepath} not found, skipping...")
            continue
        
        df = pd.read_csv(filepath)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"✅ Loaded {table_name}: {len(df):,} records")
    
    # Create indexes for better query performance
    print("\n🔍 Creating database indexes...")
    cursor = conn.cursor()
    
    indexes = [
        "CREATE INDEX idx_production_timestamp ON production(timestamp)",
        "CREATE INDEX idx_production_machine ON production(machine_id)",
        "CREATE INDEX idx_quality_timestamp ON quality(timestamp)",
        "CREATE INDEX idx_maintenance_machine ON maintenance(machine_id)",
    ]
    
    for idx_sql in indexes:
        try:
            cursor.execute(idx_sql)
            print(f"✅ Created index")
        except Exception as e:
            print(f"⚠️  Index creation warning: {e}")
    
    conn.commit()
    conn.close()
    
    print("\n✅ Database created successfully!")
    return db_path

def validate_data():
    """Validate the generated data"""
    print("\n" + "="*60)
    print("✓ VALIDATING DATA")
    print("="*60 + "\n")
    
    db_path = Path('data/manufacturing.db')
    conn = sqlite3.connect(db_path)
    
    # Check table counts
    tables = ['production', 'quality', 'maintenance', 'cnc_features', 'cnc_sensors']
    
    print("📊 Table Record Counts:")
    for table in tables:
        try:
            count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0]['count']
            print(f"   {table:20s}: {count:>10,} records")
        except Exception as e:
            print(f"   {table:20s}: ❌ Error - {e}")
    
    # Quick statistics
    print("\n📈 Production Statistics:")
    prod_stats = pd.read_sql("""
        SELECT 
            COUNT(DISTINCT machine_id) as num_machines,
            SUM(units_produced) as total_units,
            AVG(utilization_percent) as avg_utilization,
            COUNT(CASE WHEN status = 'Down' THEN 1 END) as downtime_events
        FROM production
    """, conn)
    
    print(f"   Machines: {prod_stats['num_machines'].iloc[0]}")
    print(f"   Total units produced: {prod_stats['total_units'].iloc[0]:,.0f}")
    print(f"   Average utilization: {prod_stats['avg_utilization'].iloc[0]:.1f}%")
    print(f"   Downtime events: {prod_stats['downtime_events'].iloc[0]:,}")
    
    print("\n🔍 Quality Statistics:")
    quality_stats = pd.read_sql("""
        SELECT 
            SUM(inspected_units) as total_inspected,
            SUM(defective_units) as total_defects,
            AVG(defect_rate) as avg_defect_rate
        FROM quality
    """, conn)
    
    print(f"   Total inspected: {quality_stats['total_inspected'].iloc[0]:,.0f}")
    print(f"   Total defects: {quality_stats['total_defects'].iloc[0]:,.0f}")
    print(f"   Average defect rate: {quality_stats['avg_defect_rate'].iloc[0]:.4f} ({quality_stats['avg_defect_rate'].iloc[0]*100:.2f}%)")
    
    conn.close()
    print("\n✅ Data validation complete!")

def main():
    """Main pipeline execution"""
    print("\n" + "="*70)
    print("🏭 SMART MANUFACTURING DASHBOARD - DATA PIPELINE")
    print("="*70)
    
    try:
        # Create processed directory
        processed_dir = Path('data/processed')
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Process CNC data
        print("\n[1/4] Processing CNC Tool Wear Data...")
        processor = CNCDataProcessor()
        cnc_data = processor.process_all()
        
        if cnc_data is None:
            print("\n❌ CNC data processing failed!")
            print("Make sure the dataset is unzipped in data/raw/")
            return
        
        # Save CNC data
        print("\n💾 Saving CNC processed data...")
        cnc_data['features'].to_csv(processed_dir / 'cnc_features.csv', index=False)
        cnc_data['sensor_summary'].to_csv(processed_dir / 'cnc_sensor_data.csv', index=False)
        print("✅ CNC data saved")
        
        # Step 2: Generate synthetic data
        print("\n[2/4] Generating Synthetic Manufacturing Data...")
        generator = ManufacturingDataGenerator(start_date='2024-01-01', num_days=365)
        synthetic_data = generator.generate_all_data()
        
        # Save synthetic data
        print("\n💾 Saving synthetic data...")
        synthetic_data['production'].to_csv(processed_dir / 'production_data.csv', index=False)
        synthetic_data['quality'].to_csv(processed_dir / 'quality_data.csv', index=False)
        synthetic_data['maintenance'].to_csv(processed_dir / 'maintenance_data.csv', index=False)
        print("✅ Synthetic data saved")
        
        # Step 3: Create database
        print("\n[3/4] Creating Database...")
        db_path = create_database()
        
        # Step 4: Validate
        print("\n[4/4] Validating Data...")
        validate_data()
        
        # Success!
        print("\n" + "="*70)
        print("🎉 DATA PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📂 Database location: {db_path}")
        print("\n🚀 Next steps:")
        print("   1. Build ML models: python src/models/train_models.py")
        print("   2. Launch dashboard: streamlit run dashboard/app.py")
        print("\n💡 All data is ready for the dashboard!")
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()