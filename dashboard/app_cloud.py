"""
Smart Manufacturing Dashboard - Cloud Version
Uses CSV files for data storage
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import time
import os
from typing import Dict, Any, Optional, List, Tuple

def load_csv_data(filename: str) -> pd.DataFrame:
    """Load data from a CSV file in the data/processed directory"""
    try:
        filepath = Path(f"data/processed/{filename}")
        if not filepath.exists():
            st.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        return pd.read_csv(filepath, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(filepath, nrows=1).columns else False)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return pd.DataFrame()

def generate_sample_data():
    """Generate sample data for the dashboard"""
    # Check if we already have data
    if Path("data/processed/production_data.csv").exists():
        return  # Don't regenerate if we already have data
    
    now = datetime.now()
    dates = pd.date_range(end=now, periods=30, freq='D')
    
    # Generate production data
    production_data = []
    for date in dates:
        for i in range(3):  # 3 machines
            production_data.append({
                'timestamp': date,
                'machine_id': f"CNC-{i+1}",
                'units_produced': np.random.randint(80, 200),
                'utilization_percent': np.random.uniform(60, 99.9),
                'cycle_time_minutes': np.random.uniform(1.5, 3.5),
                'status': np.random.choice(['Running', 'Idle', 'Maintenance'], p=[0.8, 0.15, 0.05]),
                'product_type': np.random.choice(['Pacemaker', 'Defibrillator', 'Catheter', 'Stent']),
                'hour': np.random.randint(0, 24)
            })
    
    # Save production data
    pd.DataFrame(production_data).to_csv("data/processed/production_data.csv", index=False)
    
    # Generate quality data
    quality_data = []
    for date in dates:
        for i in range(3):  # 3 shifts per day
            quality_data.append({
                'timestamp': date + timedelta(hours=i*8),  # 3 shifts per day
                'machine_id': f"CNC-{i+1}",
                'inspected_units': np.random.randint(100, 500),
                'defective_units': np.random.randint(0, 10),
                'defect_rate': np.random.uniform(0, 0.05),  # 0-5% defect rate
                'defect_types': ", ".join(np.random.choice(
                    ["Dimensional", "Surface Defect", "Contamination", "Assembly Error", "Packaging Issue"],
                    size=np.random.randint(1, 4),
                    replace=False
                ))
            })
    pd.DataFrame(quality_data).to_csv("data/processed/quality_data.csv", index=False)
    
    # Generate maintenance data
    maintenance_data = []
    for i in range(15):  # 15 maintenance events
        maintenance_data.append({
            'timestamp': now - timedelta(days=np.random.randint(1, 60)),
            'machine_id': f"CNC-{np.random.randint(1, 4)}",
            'maintenance_type': np.random.choice(["Preventive", "Corrective", "Predictive"]),
            'description': np.random.choice([
                "Routine inspection and cleaning",
                "Belt replacement",
                "Software update",
                "Lubrication",
                "Calibration check"
            ]),
            'duration_hours': round(np.random.uniform(0.5, 8), 1),  # 0.5 to 8 hours
            'cost_usd': round(np.random.uniform(100, 5000), 2)  # $100 to $5000
        })
    pd.DataFrame(maintenance_data).to_csv("data/processed/maintenance_data.csv", index=False)
    
    # Generate CNC features/predictions
    cnc_data = []
    for i in range(3):  # 3 machines
        cnc_data.append({
            'machine_id': f"CNC-{i+1}",
            'maintenance_risk_score': round(np.random.uniform(0, 1), 4),  # 0-1 scale
            'health_score': round(np.random.uniform(50, 100), 1)  # 50-100 scale
        })
    pd.DataFrame(cnc_data).to_csv("data/processed/cnc_features.csv", index=False)

def load_all_data():
    """Load all necessary data from CSV files"""
    data = {}
    
    try:
        # Load production data
        production_df = load_csv_data('production_data.csv')
        if not production_df.empty and 'timestamp' in production_df.columns:
            production_df['timestamp'] = pd.to_datetime(production_df['timestamp'])
            production_df['hour'] = production_df['timestamp'].dt.hour
        data['production'] = production_df
        
        # Load quality data
        quality_df = load_csv_data('quality_data.csv')
        if not quality_df.empty and 'timestamp' in quality_df.columns:
            quality_df['timestamp'] = pd.to_datetime(quality_df['timestamp'])
        data['quality'] = quality_df
        
        # Load maintenance data
        maintenance_df = load_csv_data('maintenance_data.csv')
        if not maintenance_df.empty and 'timestamp' in maintenance_df.columns:
            maintenance_df['timestamp'] = pd.to_datetime(maintenance_df['timestamp'])
        data['maintenance'] = maintenance_df
        
        # Load CNC features/predictions
        cnc_predictions = load_csv_data('cnc_features.csv')
        if not cnc_predictions.empty:
            # Ensure we have the expected columns
            if 'maintenance_risk_score' not in cnc_predictions.columns:
                cnc_predictions['maintenance_risk_score'] = np.random.uniform(0, 1, size=len(cnc_predictions))
            if 'health_score' not in cnc_predictions.columns:
                cnc_predictions['health_score'] = np.random.uniform(0, 100, size=len(cnc_predictions))
        data['cnc_predictions'] = cnc_predictions
        
        data['last_updated'] = datetime.now()
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty DataFrames with correct structure on error
        return {
            'production': pd.DataFrame(columns=['timestamp', 'machine_id', 'units_produced', 
                                             'utilization_percent', 'cycle_time_minutes', 
                                             'status', 'product_type', 'hour']),
            'quality': pd.DataFrame(columns=['timestamp', 'machine_id', 'inspected_units', 
                                          'defective_units', 'defect_rate', 'defect_types']),
            'maintenance': pd.DataFrame(columns=['timestamp', 'machine_id', 'maintenance_type', 
                                              'description', 'duration_hours', 'cost_usd']),
            'cnc_predictions': pd.DataFrame(columns=['machine_id', 'maintenance_risk_score', 'health_score']),
            'last_updated': datetime.now()
        }
    
    return data
# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Smart Manufacturing Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.loading = True
    st.session_state.data = {
        'production': pd.DataFrame(),
        'quality': pd.DataFrame(),
        'maintenance': pd.DataFrame(),
        'cnc_predictions': pd.DataFrame(),
        'quality_predictions': pd.DataFrame()
    }

# Add a status container at the top of your sidebar
with st.sidebar:
    status = st.empty()
    if st.session_state.loading:
        with status:
            with st.spinner('Loading application...'):
                time.sleep(0.5)  # Just to show the spinner
# Add to the bottom of your sidebar
with st.sidebar.expander("Debug Info", expanded=False):
    st.write("### Database Status")
    try:
        debug_conn = get_database_connection()
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", debug_conn)
        st.write("Tables in database:", tables['name'].tolist())
        
        # Show some basic stats
        for table in tables['name'].tolist():
            try:
                count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", debug_conn).iloc[0]['count']
                st.write(f"- {table}: {count} rows")
            except Exception as e:
                st.write(f"- {table}: Error - {str(e)}")
        
        debug_conn.close()
    except Exception as e:
        st.error(f"Debug error: {str(e)}")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# After data is loaded
if not st.session_state.initialized:
    data = load_all_data()
    st.session_state.data = data
    st.session_state.loading = False
    st.session_state.initialized = True
    status.empty()  # Clear the loading message


# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .alert-critical {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #f44336;
    }
    .alert-warning {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ff9800;
    }
    .alert-success {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

def ensure_data_directories():
    """Ensure required data directories exist"""
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    
    for directory in [data_dir, processed_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    return {
        'data_dir': data_dir,
        'processed_dir': processed_dir
    }

# Ensure data directory exists
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Load all data with error handling
try:
    data = load_all_data()
    
    # Ensure all required columns exist
    required_columns = {
        'production': ['timestamp', 'units_produced', 'utilization_percent', 'status', 'machine_id'],
        'quality': ['timestamp', 'defect_rate', 'inspected_units'],
        'maintenance': ['timestamp', 'machine_id', 'maintenance_type']
    }

    for table, cols in required_columns.items():
        if table in data:
            for col in cols:
                if col not in data[table].columns:
                    st.warning(f"⚠️ Missing column '{col}' in {table} data")
                    data[table][col] = 0  # or appropriate default
                    
    # Add debug information to sidebar
    st.sidebar.subheader("Debug Info")
    st.sidebar.json({
        "Production columns": list(data['production'].columns) if not data['production'].empty else [],
        "Production rows": len(data['production']),
        "Quality columns": list(data['quality'].columns) if not data['quality'].empty else [],
        "Quality rows": len(data['quality'])
    })
    
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()  # Stop execution if we can't load data


# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.loading = True
    st.session_state.data = {
        'production': pd.DataFrame(),
        'quality': pd.DataFrame(),
        'maintenance': pd.DataFrame(),
        'cnc_predictions': pd.DataFrame(),
        'quality_predictions': pd.DataFrame()
    }

# Load data if not already loaded
if not st.session_state.initialized or st.session_state.loading:
    with st.spinner('Loading application data...'):
        st.session_state.data = load_all_data()
        st.session_state.loading = False
        st.session_state.initialized = True
        
# Get data from session state
data = st.session_state.data

# Extract dataframes from the dictionary with default values
production_df = data.get('production', pd.DataFrame({
    'timestamp': [datetime.now()],
    'machine_id': ['N/A'],
    'units_produced': [0],
    'utilization_percent': [0],
    'cycle_time_minutes': [0],
    'status': ['No Data'],
    'product_type': ['N/A'],
    'hour': [0]
}))

quality_df = data.get('quality', pd.DataFrame({
    'timestamp': [datetime.now()],
    'machine_id': ['N/A'],
    'inspected_units': [0],
    'defective_units': [0],
    'defect_rate': [0],
    'defect_types': ['None']
}))

maintenance_df = data.get('maintenance', pd.DataFrame({
    'timestamp': [datetime.now()],
    'machine_id': ['N/A'],
    'maintenance_type': ['None'],
    'description': ['No maintenance records'],
    'duration_hours': [0],
    'cost_usd': [0]
}))

cnc_pred_df = data.get('cnc_predictions', pd.DataFrame({
    'machine_id': ['N/A'],
    'maintenance_risk_score': [0],
    'health_score': [0]
}))

quality_pred_df = data.get('quality_predictions', pd.DataFrame({
    'timestamp': [datetime.now()],
    'machine_id': ['N/A'],
    'anomaly_score': [0],
    'is_anomaly': [False]
}))

# Sidebar
st.sidebar.title("🏭 Smart Manufacturing")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 Production", "🔍 Quality Control", "🔧 Maintenance", "📈 Reports"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

try:
    # Get data from session state
    data = st.session_state.data
    production_df = data.get('production', pd.DataFrame())
    quality_df = data.get('quality', pd.DataFrame())
    maintenance_df = data.get('maintenance', pd.DataFrame())
    cnc_pred_df = data.get('cnc_predictions', pd.DataFrame())
    quality_pred_df = data.get('quality_predictions', pd.DataFrame())
    
    # Ensure required columns exist in the dataframes
    if not production_df.empty:
        required_prod_cols = ['units_produced', 'utilization_percent', 'cycle_time_minutes', 'status', 'product_type']
        for col in required_prod_cols:
            if col not in production_df.columns:
                production_df[col] = 0 if col != 'status' and col != 'product_type' else ''
    else:
        st.warning("No production data found in the database.")
        production_df = pd.DataFrame(columns=['timestamp', 'machine_id', 'units_produced', 'utilization_percent', 
                                           'cycle_time_minutes', 'status', 'product_type', 'hour'])
    
    if not quality_df.empty:
        required_qual_cols = ['inspected_units', 'defective_units', 'defect_rate', 'defect_types']
        for col in required_qual_cols:
            if col not in quality_df.columns:
                quality_df[col] = 0 if col != 'defect_types' else ''
    else:
        st.warning("No quality data found in the database.")
        quality_df = pd.DataFrame(columns=['timestamp', 'machine_id', 'inspected_units', 'defective_units', 
                                        'defect_rate', 'defect_types'])

except Exception as e:
    st.error(f"Error initializing application: {e}")
    st.stop()
# Set default date range
default_min_date = datetime.now().date() - timedelta(days=30)
default_max_date = datetime.now().date()

# Initialize date range variables
min_date = default_min_date
max_date = default_max_date

# Date filter - handle empty DataFrames and missing timestamp columns
# Set default date range
default_min_date = datetime.now().date() - timedelta(days=30)
default_max_date = datetime.now().date()

# Initialize date range variables
min_date = default_min_date
max_date = default_max_date

try:
    # Date filter - handle empty DataFrames and missing timestamp columns
    if not data['production'].empty and 'timestamp' in data['production'].columns:
        try:
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(data['production']['timestamp']):
                data['production']['timestamp'] = pd.to_datetime(data['production']['timestamp'], errors='coerce')
            
            # Get min/max dates, excluding NaT values
            valid_dates = data['production']['timestamp'].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
        except Exception as e:
            st.warning(f"Could not parse timestamps: {e}")
    
    # Ensure min_date is not after max_date
    if min_date > max_date:
        min_date = max_date - timedelta(days=30)

except Exception as e:
    st.error(f"Error setting date range: {e}")
    # Fall back to default dates on error
    min_date = default_min_date
    max_date = default_max_date
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
# Machine filter
machines = ['All']
if not production_df.empty and 'machine_id' in production_df.columns:
    machines += sorted(production_df['machine_id'].unique().tolist())
selected_machine = st.sidebar.selectbox("Machine", machines)

# Debug: Print input data info
print("\n=== DEBUG: Data Loading Status ===")
print(f"Production data shape: {production_df.shape if not production_df.empty else 'Empty DataFrame'}")
if not production_df.empty:
    print(f"Production columns: {production_df.columns.tolist()}")
    print(f"First row: {production_df.iloc[0].to_dict() if len(production_df) > 0 else 'No data'}")

print(f"\nQuality data shape: {quality_df.shape if not quality_df.empty else 'Empty DataFrame'}")
if not quality_df.empty:
    print(f"Quality columns: {quality_df.columns.tolist()}")

print("\n=== End of data loading ===\n")

# Filter data based on date range and machine selection
start_date = date_range[0] if date_range else min_date
end_date = date_range[1] if date_range and len(date_range) > 1 else max_date

print(f"Selected date range: {start_date} to {end_date}")
print(f"Selected machine: {selected_machine}")

# Initialize filtered data
production_filtered = pd.DataFrame()
quality_filtered = pd.DataFrame()

try:
    # Process production data
    if not production_df.empty and 'timestamp' in production_df.columns:
        production_filtered = production_df.copy()
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(production_filtered['timestamp']):
            production_filtered['timestamp'] = pd.to_datetime(production_filtered['timestamp'], errors='coerce')
    
    # Process quality data
    if not quality_df.empty and 'timestamp' in quality_df.columns:
        quality_filtered = quality_df.copy()
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(quality_filtered['timestamp']):
            quality_filtered['timestamp'] = pd.to_datetime(quality_filtered['timestamp'], errors='coerce')
    
    # Filter by machine if selected
    if selected_machine != 'All':
        print(f"Filtering by machine: {selected_machine}")
        
        # Filter production data by machine
        if not production_filtered.empty and 'machine_id' in production_filtered.columns:
            try:
                production_filtered = production_filtered[production_filtered['machine_id'] == selected_machine].copy()
                print(f"Production data after machine filter: {len(production_filtered)} rows")
            except Exception as e:
                st.error(f"Error filtering production data by machine: {e}")
        
        # Filter quality data by machine
        if not quality_filtered.empty and 'machine_id' in quality_filtered.columns:
            try:
                quality_filtered = quality_filtered[quality_filtered['machine_id'] == selected_machine].copy()
                print(f"Quality data after machine filter: {len(quality_filtered)} rows")
            except Exception as e:
                st.error(f"Error filtering quality data by machine: {e}")

except Exception as e:
    st.error(f"Error processing data: {e}")
    # If there's an error, fall back to empty DataFrames
    production_filtered = pd.DataFrame()
    quality_filtered = pd.DataFrame()
    
    # Ensure required columns exist with default values if missing
    if not production_filtered.empty:
        required_cols = {
            'units_produced': 0,
            'utilization_percent': 0.0,
            'cycle_time_minutes': 0.0,
            'status': 'Unknown',
            'product_type': 'Unknown',
            'hour': 0
        }
        for col, default in required_cols.items():
            if col not in production_filtered.columns:
                print(f"Warning: Adding missing column '{col}' with default value: {default}")
                production_filtered[col] = default
    
    print("=== End of data filtering ===\n")
    
except Exception as e:
    st.error(f"Error processing data: {e}")
    st.error("Please check the debug info in the sidebar for more details.")
    st.stop()

# ==============================================================================
# PAGE: OVERVIEW
# ==============================================================================
if page == "🏠 Overview":
    st.title("🏠 Manufacturing Overview Dashboard")
    st.markdown("Real-time monitoring of production, quality, and equipment health")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_units = production_filtered['units_produced'].sum()
        st.metric(
            label="📦 Total Units Produced",
            value=f"{total_units:,.0f}",
            delta=f"+{(total_units / len(production_filtered) * 24):.0f} per day"
        )
    
    with col2:
        avg_utilization = production_filtered['utilization_percent'].mean()
        st.metric(
            label="⚙️ Avg Utilization",
            value=f"{avg_utilization:.1f}%",
            delta=f"{avg_utilization - 90:.1f}%" if avg_utilization >= 90 else f"{avg_utilization - 90:.1f}%"
        )
    
    with col3:
        if not quality_filtered.empty and 'defect_rate' in quality_filtered.columns:
            defect_rate = quality_filtered['defect_rate'].mean() * 100
            st.metric(
                label="🎯 Defect Rate",
                value=f"{defect_rate:.3f}%",
                delta=f"{defect_rate - 0.25:.3f}%" if defect_rate < 0.25 else f"{defect_rate - 0.25:.3f}%"
            )
        else:
            st.metric(
                label="🎯 Defect Rate",
                value="N/A",
                help="No quality data available"
            )
    
    with col4:
        downtime_events = len(production_filtered[production_filtered['status'] == 'Down'])
        st.metric(
            label="⚠️ Downtime Events",
            value=f"{downtime_events}",
            delta=f"-{downtime_events // 10}" if downtime_events > 0 else "0"
        )
    
    st.markdown("---")
    
    # Alerts Section
    st.subheader("🚨 Active Alerts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Maintenance alerts
        if not cnc_pred_df.empty and 'maintenance_risk_score' in cnc_pred_df.columns and 'machine_id' in cnc_pred_df.columns:
            try:
                high_risk_machines = cnc_pred_df[cnc_pred_df['maintenance_risk_score'] > 70]
                if len(high_risk_machines) > 0:
                    st.markdown('<div class="alert-critical">', unsafe_allow_html=True)
                    st.markdown("**⚠️ CRITICAL: Maintenance Required**")
                    for _, machine in high_risk_machines.iterrows():
                        st.markdown(f"- {machine['machine_id']}: Risk Score {machine['maintenance_risk_score']:.0f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-success">', unsafe_allow_html=True)
                    st.markdown("**✅ All machines healthy**")
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown('<div class="alert-warning">', unsafe_allow_html=True)
                st.markdown("**⚠️ Unable to load maintenance data**")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-warning">', unsafe_allow_html=True)
            st.markdown("**ℹ️ No maintenance data available**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Quality alerts - filter by date range and machine
        quality_pred_filtered = quality_pred_df.copy()
        
        # Ensure we have a timestamp column to work with
        if 'timestamp' in quality_pred_filtered.columns:
            quality_pred_filtered['timestamp'] = pd.to_datetime(quality_pred_filtered['timestamp'])
            
            if len(date_range) == 2:
                mask = (quality_pred_filtered['timestamp'].dt.date >= date_range[0]) & \
                       (quality_pred_filtered['timestamp'].dt.date <= date_range[1])
                quality_pred_filtered = quality_pred_filtered[mask]
            
            if selected_machine != 'All' and 'machine_id' in quality_pred_filtered.columns:
                quality_pred_filtered = quality_pred_filtered[quality_pred_filtered['machine_id'] == selected_machine]
            
            recent_anomalies = quality_pred_filtered[quality_pred_filtered.get('is_anomaly', False) == True].tail(5)
        else:
            recent_anomalies = pd.DataFrame()  # Empty DataFrame if no timestamp column
        
        if len(recent_anomalies) > 0:
            st.markdown('<div class="alert-warning">', unsafe_allow_html=True)
            st.markdown("**⚠️ Quality Anomalies Detected**")
            for _, anom in recent_anomalies.iterrows():
                st.markdown(f"- {anom['machine_id']} on {anom['date']}: Anomaly Score {anom['anomaly_score']:.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-success">', unsafe_allow_html=True)
            st.markdown("**✅ Quality within normal range**")
            st.markdown(f"No anomalies detected in selected period ({len(quality_pred_filtered)} days checked)")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Daily Production Trend")
        daily_production = production_filtered.groupby(
            production_filtered['timestamp'].dt.date
        )['units_produced'].sum().reset_index()
        daily_production.columns = ['date', 'units']
        
        fig = px.line(
            daily_production, 
            x='date', 
            y='units',
            title='Units Produced per Day'
        )
        fig.update_traces(line_color='#1f77b4', line_width=3)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Quality Trend")
        if not quality_filtered.empty and 'timestamp' in quality_filtered.columns and 'defect_rate' in quality_filtered.columns:
            try:
                daily_quality = quality_filtered.groupby(
                    quality_filtered['timestamp'].dt.date
                ).agg({
                    'defect_rate': 'mean',
                    'inspected_units': 'sum' if 'inspected_units' in quality_filtered.columns else None
                }).reset_index()
                
                fig = px.line(
                    daily_quality,
                    x='timestamp',
                    y='defect_rate',
                    title='Daily Defect Rate Trend',
                    labels={'defect_rate': 'Defect Rate (%)', 'timestamp': 'Date'},
                    height=300
                )
                fig.update_traces(line=dict(color='#FF4B4B', width=2))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Could not generate quality trend chart")
        else:
            st.info("No quality data available for trend analysis")
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏭 Machine Utilization")
        machine_util = production_filtered.groupby('machine_id')['utilization_percent'].mean().reset_index()
        machine_util = machine_util.sort_values('utilization_percent', ascending=True)
        
        fig = px.bar(
            machine_util,
            x='utilization_percent',
            y='machine_id',
            orientation='h',
            title='Average Utilization by Machine',
            color='utilization_percent',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📦 Production by Product Type")
        product_mix = production_filtered.groupby('product_type')['units_produced'].sum().reset_index()
        
        fig = px.pie(
            product_mix,
            values='units_produced',
            names='product_type',
            title='Product Mix Distribution'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE: PRODUCTION
# ==============================================================================
elif page == "📊 Production":
    st.title("📊 Production Analytics")
    
    # Production metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_units = production_filtered['units_produced'].sum()
        st.metric("Total Units", f"{total_units:,.0f}")
    
    with col2:
        avg_cycle_time = production_filtered['cycle_time_minutes'].mean()
        st.metric("Avg Cycle Time", f"{avg_cycle_time:.2f} min")
    
    with col3:
        oee = production_filtered['utilization_percent'].mean()
        st.metric("OEE", f"{oee:.1f}%")
    
    with col4:
        running_time = len(production_filtered[production_filtered['status'] == 'Running'])
        total_time = len(production_filtered)
        uptime = (running_time / total_time * 100) if total_time > 0 else 0
        st.metric("Uptime", f"{uptime:.1f}%")
    
    st.markdown("---")
    
    # Hourly production heatmap
    st.subheader("📅 Production Heatmap (Hour x Day)")
    
    heatmap_data = production_filtered.pivot_table(
        values='units_produced',
        index='hour',
        columns=production_filtered['timestamp'].dt.date,
        aggfunc='sum',
        fill_value=0
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[str(d) for d in heatmap_data.columns],
        y=heatmap_data.index,
        colorscale='Blues',
        hoverongaps=False
    ))
    fig.update_layout(
        title='Production Output by Hour and Day',
        xaxis_title='Date',
        yaxis_title='Hour of Day',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE: QUALITY CONTROL
# ==============================================================================
elif page == "🔍 Quality Control":
    st.title("🔍 Quality Control Dashboard")
    
    # Quality metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_inspected = quality_filtered['inspected_units'].sum() if not quality_filtered.empty and 'inspected_units' in quality_filtered.columns else 0
        st.metric("Units Inspected", f"{total_inspected:,.0f}")
    
    with col2:
        total_defects = quality_filtered['defective_units'].sum() if not quality_filtered.empty and 'defective_units' in quality_filtered.columns else 0
        st.metric("Total Defects", f"{total_defects:,.0f}")
    
    with col3:
        if not quality_filtered.empty and 'defect_rate' in quality_filtered.columns:
            avg_defect_rate = quality_filtered['defect_rate'].mean() * 100
            st.metric("Avg Defect Rate", f"{avg_defect_rate:.3f}%")
        else:
            st.metric("Avg Defect Rate", "N/A")
    
    with col4:
        anomalies = quality_pred_df['is_anomaly'].sum()
        st.metric("Anomalies Detected", f"{anomalies}")
    
    st.markdown("---")
    
    # Control chart
    st.subheader("📊 Statistical Process Control Chart")
    
    if not quality_filtered.empty and 'timestamp' in quality_filtered.columns and 'defect_rate' in quality_filtered.columns:
        try:
            daily_defects = quality_filtered.groupby(
                quality_filtered['timestamp'].dt.date
            )['defect_rate'].mean().reset_index()
            daily_defects.columns = ['date', 'defect_rate']
            daily_defects['defect_rate'] *= 100
        except Exception as e:
            st.warning("Could not generate SPC chart: " + str(e))
            daily_defects = pd.DataFrame(columns=['date', 'defect_rate'])
    else:
        st.info("No quality data available for SPC chart")
        daily_defects = pd.DataFrame(columns=['date', 'defect_rate'])
    
    # Calculate control limits
    mean_defect = daily_defects['defect_rate'].mean()
    std_defect = daily_defects['defect_rate'].std()
    ucl = mean_defect + 3 * std_defect
    lcl = max(0, mean_defect - 3 * std_defect)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_defects['date'],
        y=daily_defects['defect_rate'],
        mode='lines+markers',
        name='Defect Rate',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_hline(y=mean_defect, line_dash="dash", line_color="green", 
                  annotation_text=f"Mean: {mean_defect:.3f}%")
    fig.add_hline(y=ucl, line_dash="dot", line_color="red",
                  annotation_text=f"UCL: {ucl:.3f}%")
    fig.add_hline(y=lcl, line_dash="dot", line_color="red",
                  annotation_text=f"LCL: {lcl:.3f}%")
    
    fig.update_layout(
        title='Defect Rate Control Chart',
        xaxis_title='Date',
        yaxis_title='Defect Rate (%)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE: MAINTENANCE
# ==============================================================================
elif page == "🔧 Maintenance":
    st.title("🔧 Predictive Maintenance Dashboard")
    
    # Maintenance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_maintenance = len(maintenance_df)
        st.metric("Total Maintenance Events", f"{total_maintenance}")
    
    with col2:
        scheduled = len(maintenance_df[maintenance_df['maintenance_type'] == 'Scheduled'])
        st.metric("Scheduled", f"{scheduled}")
    
    with col3:
        unscheduled = len(maintenance_df[maintenance_df['maintenance_type'] == 'Unscheduled'])
        st.metric("Unscheduled", f"{unscheduled}")
    
    with col4:
        total_cost = maintenance_df['cost_usd'].sum()
        st.metric("Total Cost", f"${total_cost:,.0f}")
    
    st.markdown("---")
    
    # Predictive maintenance alerts
    st.subheader("🚨 Machine Health Status")
    
    # Aggregate by machine to get unique machines with max risk
    cnc_pred_aggregated = cnc_pred_df.groupby('machine_id').agg({
        'maintenance_risk_score': 'max',
        'health_score': 'min'
    }).reset_index()
    
    # Display machine health
    for _, machine in cnc_pred_aggregated.iterrows():
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.markdown(f"### {machine['machine_id']}")
        
        with col2:
            risk = machine['maintenance_risk_score']
            health = machine['health_score']
            
            if risk > 70:
                color = "🔴"
                status = "CRITICAL"
            elif risk > 40:
                color = "🟡"
                status = "WARNING"
            else:
                color = "🟢"
                status = "HEALTHY"
            
            st.markdown(f"{color} **{status}** - Health: {health:.0f}% | Risk: {risk:.0f}%")
            st.progress(risk / 100)
        
        with col3:
            if risk > 70:
                st.button("Schedule Maintenance", key=f"maint_{machine['machine_id']}")

# ==============================================================================
# PAGE: REPORTS
# ==============================================================================
elif page == "📈 Reports":
    st.title("📈 Executive Reports")
    
    st.subheader("📊 Summary Statistics")
    
    # Overall summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Production Summary")
        summary_stats = {
            "Total Units Produced": f"{production_filtered['units_produced'].sum():,.0f}",
            "Average Utilization": f"{production_filtered['utilization_percent'].mean():.1f}%",
            "Total Operating Hours": f"{len(production_filtered):,.0f}",
            "Number of Machines": f"{production_filtered['machine_id'].nunique()}",
        }
        
        for key, value in summary_stats.items():
            st.metric(key, value)
    
    with col2:
        st.markdown("### Quality Summary")
        quality_stats = {
            "Total Inspected": f"{quality_filtered['inspected_units'].sum() if not quality_filtered.empty and 'inspected_units' in quality_filtered.columns else 0:,.0f}",
            "Total Defects": f"{quality_filtered['defective_units'].sum() if not quality_filtered.empty and 'defective_units' in quality_filtered.columns else 0:,.0f}",
            "Average Defect Rate": f"{quality_filtered['defect_rate'].mean()*100:.3f}%" if not quality_filtered.empty and 'defect_rate' in quality_filtered.columns else "N/A",
            "Anomalies Detected": f"{quality_pred_df['is_anomaly'].sum() if not quality_pred_df.empty and 'is_anomaly' in quality_pred_df.columns else 0}",
        }
        
        for key, value in quality_stats.items():
            st.metric(key, value)
    
    st.markdown("---")
    
    # Download reports
    st.subheader("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_production = production_filtered.to_csv(index=False)
        st.download_button(
            label="📊 Download Production Data",
            data=csv_production,
            file_name="production_data.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_quality = quality_filtered.to_csv(index=False)
        st.download_button(
            label="🔍 Download Quality Data",
            data=csv_quality,
            file_name="quality_data.csv",
            mime="text/csv"
        )
    
    with col3:
        csv_maintenance = maintenance_df.to_csv(index=False)
        st.download_button(
            label="🔧 Download Maintenance Data",
            data=csv_maintenance,
            file_name="maintenance_data.csv",
            mime="text/csv"
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    **Smart Manufacturing Dashboard**
    
    AI-powered monitoring system for medical device manufacturing.
    
    Features:
    - Real-time production monitoring
    - Predictive maintenance
    - Quality anomaly detection
    - Automated reporting
    
    Built with Streamlit, Plotly, and scikit-learn
    """
)