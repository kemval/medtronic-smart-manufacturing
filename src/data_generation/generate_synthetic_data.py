"""
Synthetic Manufacturing Data Generator
Generates realistic production, quality, and maintenance data
Save as: src/data_generation/generate_synthetic_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

class ManufacturingDataGenerator:
    """Generate synthetic manufacturing data for medical device production"""
    
    def __init__(self, start_date='2024-01-01', num_days=365):
        self.start_date = pd.to_datetime(start_date)
        self.num_days = num_days
        self.machines = ['CNC_A01', 'CNC_A02', 'CNC_B01', 'CNC_B02', 'CNC_C01']
        self.shifts = ['Morning', 'Afternoon', 'Night']
        self.product_types = ['Valve_Component', 'Pump_Housing', 'Surgical_Tool', 'Implant_Part']
        
    def generate_production_data(self):
        """Generate hourly production data"""
        print("📊 Generating production data...")
        
        data = []
        current_date = self.start_date
        
        for day in range(self.num_days):
            for hour in range(24):
                timestamp = current_date + timedelta(days=day, hours=hour)
                
                # Determine shift
                if 6 <= hour < 14:
                    shift = 'Morning'
                    base_production = 45
                elif 14 <= hour < 22:
                    shift = 'Afternoon'
                    base_production = 42
                else:
                    shift = 'Night'
                    base_production = 35
                
                # Weekend reduction
                if timestamp.weekday() >= 5:
                    base_production *= 0.7
                
                for machine in self.machines:
                    # Random variation
                    units_produced = int(base_production * np.random.normal(1.0, 0.15))
                    units_produced = max(0, units_produced)
                    
                    # Occasional downtime
                    if random.random() < 0.02:  # 2% chance of downtime
                        units_produced = 0
                        status = 'Down'
                    elif random.random() < 0.05:  # 5% chance of reduced performance
                        units_produced = int(units_produced * 0.5)
                        status = 'Degraded'
                    else:
                        status = 'Running'
                    
                    # Cycle time (inverse of production rate)
                    if units_produced > 0:
                        cycle_time = round(60 / (units_produced / 60), 2)  # minutes per unit
                    else:
                        cycle_time = None
                    
                    # Utilization percentage
                    utilization = round((units_produced / base_production) * 100, 1) if base_production > 0 else 0
                    
                    # Product type (weighted random)
                    product = random.choices(
                        self.product_types,
                        weights=[0.4, 0.3, 0.2, 0.1],
                        k=1
                    )[0]
                    
                    data.append({
                        'timestamp': timestamp,
                        'date': timestamp.date(),
                        'hour': hour,
                        'shift': shift,
                        'machine_id': machine,
                        'product_type': product,
                        'units_produced': units_produced,
                        'cycle_time_minutes': cycle_time,
                        'utilization_percent': utilization,
                        'status': status
                    })
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df):,} production records")
        return df
    
    def generate_quality_data(self, production_df):
        """Generate quality control data based on production"""
        print("🔍 Generating quality control data...")
        
        data = []
        
        for idx, prod_row in production_df.iterrows():
            if prod_row['units_produced'] == 0:
                continue
            
            # Sample ~10% of production for inspection
            sample_size = max(1, int(prod_row['units_produced'] * 0.1))
            
            # Base defect rate (medical devices = very low!)
            base_defect_rate = 0.002  # 0.2% base rate
            
            # Machine-specific defect rates
            machine_factor = {
                'CNC_A01': 1.0,
                'CNC_A02': 1.2,  # Slightly worse
                'CNC_B01': 0.8,
                'CNC_B02': 1.1,
                'CNC_C01': 1.5   # Needs maintenance
            }.get(prod_row['machine_id'], 1.0)
            
            # Status affects quality
            status_factor = {
                'Running': 1.0,
                'Degraded': 3.0,  # More defects when degraded
                'Down': 0
            }.get(prod_row['status'], 1.0)
            
            defect_rate = base_defect_rate * machine_factor * status_factor
            
            # Inject some anomalies (quality issues)
            if random.random() < 0.01:  # 1% chance of quality incident
                defect_rate *= 10
            
            # Generate defects
            num_defects = np.random.binomial(sample_size, defect_rate)
            
            # Defect types
            defect_types = []
            if num_defects > 0:
                defect_categories = [
                    'Dimensional_Out_Of_Spec',
                    'Surface_Finish_Issue',
                    'Burr_Present',
                    'Tool_Mark',
                    'Material_Defect'
                ]
                defect_types = random.choices(defect_categories, k=num_defects)
            
            data.append({
                'timestamp': prod_row['timestamp'],
                'date': prod_row['date'],
                'machine_id': prod_row['machine_id'],
                'product_type': prod_row['product_type'],
                'shift': prod_row['shift'],
                'inspected_units': sample_size,
                'defective_units': num_defects,
                'defect_rate': round(num_defects / sample_size, 4) if sample_size > 0 else 0,
                'defect_types': ','.join(defect_types) if defect_types else None,
                'inspector_id': f"QC_{random.randint(1, 8):02d}"
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df):,} quality inspection records")
        return df
    
    def generate_maintenance_data(self):
        """Generate maintenance logs"""
        print("🔧 Generating maintenance data...")
        
        data = []
        current_date = self.start_date
        
        for machine in self.machines:
            # Scheduled maintenance every 30 days
            for month in range(self.num_days // 30):
                maint_date = current_date + timedelta(days=month * 30 + random.randint(0, 5))
                
                data.append({
                    'timestamp': maint_date,
                    'date': maint_date.date(),
                    'machine_id': machine,
                    'maintenance_type': 'Scheduled',
                    'description': 'Routine preventive maintenance',
                    'duration_hours': round(random.uniform(2, 4), 1),
                    'parts_replaced': random.choice([None, 'Cutting Tool', 'Lubricant', 'Filter']),
                    'technician_id': f"TECH_{random.randint(1, 5):02d}",
                    'cost_usd': round(random.uniform(200, 800), 2)
                })
            
            # Unscheduled maintenance (breakdowns)
            num_breakdowns = random.randint(2, 8)
            for _ in range(num_breakdowns):
                breakdown_date = current_date + timedelta(days=random.randint(0, self.num_days))
                
                issues = [
                    ('Spindle bearing failure', 6, 1500),
                    ('Coolant pump failure', 3, 400),
                    ('Tool changer malfunction', 4, 800),
                    ('Control system error', 2, 300),
                    ('Hydraulic leak', 5, 600)
                ]
                issue = random.choice(issues)
                
                data.append({
                    'timestamp': breakdown_date,
                    'date': breakdown_date.date(),
                    'machine_id': machine,
                    'maintenance_type': 'Unscheduled',
                    'description': issue[0],
                    'duration_hours': issue[1] + random.uniform(-1, 2),
                    'parts_replaced': issue[0].split()[0],
                    'technician_id': f"TECH_{random.randint(1, 5):02d}",
                    'cost_usd': issue[2] + random.uniform(-100, 200)
                })
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        print(f"✅ Generated {len(df):,} maintenance records")
        return df
    
    def generate_all_data(self):
        """Generate all datasets"""
        print("\n" + "="*60)
        print("🏭 GENERATING SYNTHETIC MANUFACTURING DATA")
        print("="*60 + "\n")
        
        production_df = self.generate_production_data()
        quality_df = self.generate_quality_data(production_df)
        maintenance_df = self.generate_maintenance_data()
        
        return {
            'production': production_df,
            'quality': quality_df,
            'maintenance': maintenance_df
        }

def main():
    """Main execution"""
    # Create output directory
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    generator = ManufacturingDataGenerator(start_date='2024-01-01', num_days=365)
    datasets = generator.generate_all_data()
    
    # Save to CSV
    print("\n💾 Saving datasets...")
    for name, df in datasets.items():
        filepath = output_dir / f'{name}_data.csv'
        df.to_csv(filepath, index=False)
        print(f"✅ Saved: {filepath}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("📊 DATA GENERATION SUMMARY")
    print("="*60)
    print(f"Production records: {len(datasets['production']):,}")
    print(f"Quality records: {len(datasets['quality']):,}")
    print(f"Maintenance records: {len(datasets['maintenance']):,}")
    print(f"\nDate range: {datasets['production']['date'].min()} to {datasets['production']['date'].max()}")
    print(f"Machines: {', '.join(generator.machines)}")
    print(f"Product types: {', '.join(generator.product_types)}")
    print("\n✅ All synthetic data generated successfully!")

if __name__ == "__main__":
    main()