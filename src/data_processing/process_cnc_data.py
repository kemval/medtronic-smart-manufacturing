"""
CNC Tool Wear Data Processor
Processes the real CNC dataset from Kaggle
Save as: src/data_processing/process_cnc_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

class CNCDataProcessor:
    """Process CNC tool wear dataset"""
    
    def __init__(self, raw_data_path='data/raw'):
        self.raw_data_path = Path(raw_data_path)
        self.machines = ['CNC_A01', 'CNC_A02', 'CNC_B01', 'CNC_B02', 'CNC_C01']
    
    def load_all_experiments(self):
        """Load all experiment CSV files"""
        print("📂 Loading CNC experiment data...")
        
        csv_files = sorted(glob.glob(str(self.raw_data_path / 'experiment_*.csv')))
        
        if not csv_files:
            print("❌ No experiment CSV files found in data/raw/")
            print("   Make sure you've unzipped the dataset to data/raw/")
            return None
        
        print(f"   Found {len(csv_files)} experiment files")
        
        all_data = []
        for idx, filepath in enumerate(csv_files, 1):
            try:
                df = pd.read_csv(filepath)
                df['experiment_id'] = idx
                df['machine_id'] = np.random.choice(self.machines)  # Assign to random machine
                all_data.append(df)
                print(f"   ✅ Loaded experiment {idx:02d}: {len(df):,} rows")
            except Exception as e:
                print(f"   ⚠️  Error loading {filepath}: {e}")
        
        if not all_data:
            return None
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ Total CNC records loaded: {len(combined_df):,}")
        return combined_df
    
    def extract_features(self, df):
        """Extract key features for ML models"""
        print("\n🔧 Extracting features from CNC data...")
        
        # Key columns for tool wear prediction
        feature_cols = [
            'X1_ActualVelocity', 'Y1_ActualVelocity', 'Z1_ActualVelocity',
            'S1_ActualVelocity',  # Spindle speed
            'X1_CurrentFeedback', 'Y1_CurrentFeedback', 'Z1_CurrentFeedback',
            'S1_CurrentFeedback',  # Motor currents
        ]
        
        # Check which columns exist
        available_cols = [col for col in feature_cols if col in df.columns]
        print(f"   Available sensor columns: {len(available_cols)}")
        
        if not available_cols:
            print("   ⚠️  Warning: No expected sensor columns found!")
            return df
        
        # Create aggregated features by experiment and machine
        features = df.groupby(['experiment_id', 'machine_id']).agg({
            **{col: ['mean', 'std', 'max', 'min'] for col in available_cols}
        }).reset_index()
        
        # Flatten column names
        features.columns = ['_'.join(col).strip('_') for col in features.columns.values]
        
        # Add operating hours (proxy for tool wear)
        features['operating_hours'] = df.groupby(['experiment_id', 'machine_id']).size().values / 100
        
        # Create tool wear indicator (synthetic, based on operating hours)
        features['tool_wear_level'] = (
            features['operating_hours'] / features['operating_hours'].max()
        ) * 100
        
        # Add vibration proxy (std of velocities)
        velocity_cols = [col for col in features.columns if 'ActualVelocity_std' in col]
        if velocity_cols:
            features['vibration_index'] = features[velocity_cols].mean(axis=1)
        
        # Health score (0-100, inverse of tool wear)
        features['health_score'] = 100 - features['tool_wear_level']
        
        print(f"✅ Extracted features: {len(features):,} records with {len(features.columns)} features")
        return features
    
    def create_sensor_summary(self, df):
        """Create time-series summary for dashboard"""
        print("\n📊 Creating sensor time-series summary...")
        
        # Sample data to manageable size for dashboard
        sample_size = min(10000, len(df))
        sampled_df = df.sample(n=sample_size, random_state=42).copy()
        
        # Add timestamp (synthetic)
        start_date = pd.Timestamp('2024-01-01')
        sampled_df['timestamp'] = [
            start_date + pd.Timedelta(minutes=i*5) 
            for i in range(len(sampled_df))
        ]
        sampled_df = sampled_df.sort_values('timestamp')
        
        # Keep key columns
        cols_to_keep = ['timestamp', 'machine_id', 'experiment_id']
        sensor_cols = [col for col in sampled_df.columns if 'Current' in col or 'Velocity' in col]
        cols_to_keep.extend(sensor_cols[:8])  # Limit to 8 sensor columns
        
        summary_df = sampled_df[cols_to_keep].copy()
        
        print(f"✅ Created sensor summary: {len(summary_df):,} records")
        return summary_df
    
    def process_all(self):
        """Process all CNC data"""
        print("\n" + "="*60)
        print("⚙️  PROCESSING CNC TOOL WEAR DATA")
        print("="*60 + "\n")
        
        # Load data
        raw_df = self.load_all_experiments()
        if raw_df is None:
            return None
        
        # Extract features for ML
        features_df = self.extract_features(raw_df)
        
        # Create sensor summary for dashboard
        sensor_df = self.create_sensor_summary(raw_df)
        
        return {
            'raw': raw_df,
            'features': features_df,
            'sensor_summary': sensor_df
        }

def main():
    """Main execution"""
    # Create output directory
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process data
    processor = CNCDataProcessor()
    datasets = processor.process_all()
    
    if datasets is None:
        print("\n❌ Failed to process CNC data. Check that CSV files are in data/raw/")
        return
    
    # Save processed data
    print("\n💾 Saving processed datasets...")
    
    # Save features for ML models
    features_path = output_dir / 'cnc_features.csv'
    datasets['features'].to_csv(features_path, index=False)
    print(f"✅ Saved: {features_path}")
    
    # Save sensor summary for dashboard
    sensor_path = output_dir / 'cnc_sensor_data.csv'
    datasets['sensor_summary'].to_csv(sensor_path, index=False)
    print(f"✅ Saved: {sensor_path}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 CNC DATA PROCESSING SUMMARY")
    print("="*60)
    print(f"Raw records processed: {len(datasets['raw']):,}")
    print(f"Feature records: {len(datasets['features']):,}")
    print(f"Sensor time-series records: {len(datasets['sensor_summary']):,}")
    print("\n✅ CNC data processed successfully!")

if __name__ == "__main__":
    main()