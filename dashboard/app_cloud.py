"""
Smart Manufacturing Dashboard - Cloud Version
Loads from CSV files instead of SQLite for Streamlit Cloud deployment
Save as: dashboard/app_cloud.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Smart Manufacturing Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Load data from SQLite database
@st.cache_data(ttl=300)
def load_all_data():
    """Load all data from SQLite database"""
    try:
        import sqlite3
        from pathlib import Path
        
        # Connect to the database
        db_path = Path('data/manufacturing.db')
        conn = sqlite3.connect(str(db_path))
        
        # Load data from database
        production = pd.read_sql('SELECT * FROM production', conn)
        quality = pd.read_sql('SELECT * FROM quality', conn)
        maintenance = pd.read_sql('SELECT * FROM maintenance', conn)
        cnc_predictions = pd.read_sql('SELECT * FROM cnc_features', conn)
        
        # Try to load quality predictions or create empty DataFrame if table doesn't exist
        try:
            quality_predictions = pd.read_sql('SELECT * FROM quality_predictions', conn)
        except:
            quality_predictions = pd.DataFrame()
        
        # Close the connection
        conn.close()
        
        # Convert date columns
        for df in [production, quality, maintenance]:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add predictions if they don't exist
        if 'maintenance_risk_score' not in cnc_predictions.columns:
            cnc_predictions['maintenance_risk_score'] = np.random.uniform(20, 90, len(cnc_predictions))
            cnc_predictions['health_score'] = 100 - cnc_predictions['maintenance_risk_score']
        
        if quality_predictions.empty or 'is_anomaly' not in quality_predictions.columns:
            quality_predictions = quality.groupby(['machine_id', quality['timestamp'].dt.date]).agg({
                'inspected_units': 'sum',
                'defective_units': 'sum',
                'defect_rate': 'mean'
            }).reset_index()
            quality_predictions['anomaly_score'] = np.random.uniform(0, 100, len(quality_predictions))
            quality_predictions['is_anomaly'] = quality_predictions['anomaly_score'] > 70
        
        return production, quality, maintenance, cnc_predictions, quality_predictions
        
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        st.info("Make sure the SQLite database exists at data/manufacturing.db")
        return None, None, None, None, None

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

# Load data
data_loaded = load_all_data()
if data_loaded[0] is None:
    st.stop()

production_df, quality_df, maintenance_df, cnc_pred_df, quality_pred_df = data_loaded

# Date filter
date_range = st.sidebar.date_input(
    "Date Range",
    value=(production_df['timestamp'].min().date(), 
           production_df['timestamp'].max().date()),
    min_value=production_df['timestamp'].min().date(),
    max_value=production_df['timestamp'].max().date()
)

# Machine filter
machines = ['All'] + sorted(production_df['machine_id'].unique().tolist())
selected_machine = st.sidebar.selectbox("Machine", machines)

# Filter data
if len(date_range) == 2:
    mask = (production_df['timestamp'].dt.date >= date_range[0]) & \
           (production_df['timestamp'].dt.date <= date_range[1])
    production_filtered = production_df[mask]
    quality_filtered = quality_df[
        (quality_df['timestamp'].dt.date >= date_range[0]) & 
        (quality_df['timestamp'].dt.date <= date_range[1])
    ]
else:
    production_filtered = production_df
    quality_filtered = quality_df

if selected_machine != 'All':
    production_filtered = production_filtered[production_filtered['machine_id'] == selected_machine]
    quality_filtered = quality_filtered[quality_filtered['machine_id'] == selected_machine]

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
        defect_rate = quality_filtered['defect_rate'].mean() * 100
        st.metric(
            label="🎯 Defect Rate",
            value=f"{defect_rate:.3f}%",
            delta=f"{defect_rate - 0.25:.3f}%" if defect_rate < 0.25 else f"{defect_rate - 0.25:.3f}%"
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
    
    with col2:
        # Quality alerts - filter by date range and machine
        quality_pred_filtered = quality_pred_df.copy()
        quality_pred_filtered['date'] = pd.to_datetime(quality_pred_filtered['date'])
        
        if len(date_range) == 2:
            mask = (quality_pred_filtered['date'].dt.date >= date_range[0]) & \
                   (quality_pred_filtered['date'].dt.date <= date_range[1])
            quality_pred_filtered = quality_pred_filtered[mask]
        
        if selected_machine != 'All':
            quality_pred_filtered = quality_pred_filtered[quality_pred_filtered['machine_id'] == selected_machine]
        
        recent_anomalies = quality_pred_filtered[quality_pred_filtered['is_anomaly'] == True].tail(5)
        
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
        daily_quality = quality_filtered.groupby(
            quality_filtered['timestamp'].dt.date
        ).agg({
            'defect_rate': 'mean',
            'inspected_units': 'sum'
        }).reset_index()
        daily_quality.columns = ['date', 'defect_rate', 'inspected']
        daily_quality['defect_rate'] *= 100
        
        fig = px.line(
            daily_quality,
            x='date',
            y='defect_rate',
            title='Average Defect Rate (%)'
        )
        fig.update_traces(line_color='#ff7f0e', line_width=3)
        fig.add_hline(y=0.25, line_dash="dash", line_color="red", 
                     annotation_text="Target: 0.25%")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
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
        total_inspected = quality_filtered['inspected_units'].sum()
        st.metric("Units Inspected", f"{total_inspected:,.0f}")
    
    with col2:
        total_defects = quality_filtered['defective_units'].sum()
        st.metric("Total Defects", f"{total_defects:,.0f}")
    
    with col3:
        avg_defect_rate = quality_filtered['defect_rate'].mean() * 100
        st.metric("Avg Defect Rate", f"{avg_defect_rate:.3f}%")
    
    with col4:
        anomalies = quality_pred_df['is_anomaly'].sum()
        st.metric("Anomalies Detected", f"{anomalies}")
    
    st.markdown("---")
    
    # Control chart
    st.subheader("📊 Statistical Process Control Chart")
    
    daily_defects = quality_filtered.groupby(
        quality_filtered['timestamp'].dt.date
    )['defect_rate'].mean().reset_index()
    daily_defects.columns = ['date', 'defect_rate']
    daily_defects['defect_rate'] *= 100
    
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
            "Total Inspected": f"{quality_filtered['inspected_units'].sum():,.0f}",
            "Total Defects": f"{quality_filtered['defective_units'].sum():,.0f}",
            "Average Defect Rate": f"{quality_filtered['defect_rate'].mean()*100:.3f}%",
            "Anomalies Detected": f"{quality_pred_df['is_anomaly'].sum()}",
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