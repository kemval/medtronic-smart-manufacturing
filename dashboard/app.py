"""
Smart Manufacturing Dashboard
Main Streamlit Application
Save as: dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
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

# Database connection
@st.cache_resource
def get_database_connection():
    """Get database connection"""
    db_path = Path('data/manufacturing.db')
    return sqlite3.connect(db_path, check_same_thread=False)

@st.cache_data(ttl=300)
def load_data(query):
    """Load data from database with caching"""
    conn = get_database_connection()
    return pd.read_sql(query, conn)

# Load all data
@st.cache_data(ttl=300)
def load_all_data():
    """Load all necessary data"""
    production = load_data("SELECT * FROM production")
    quality = load_data("SELECT * FROM quality")
    maintenance = load_data("SELECT * FROM maintenance")
    cnc_predictions = load_data("SELECT * FROM cnc_predictions")
    quality_predictions = load_data("SELECT * FROM quality_predictions")
    
    # Convert date columns
    production['timestamp'] = pd.to_datetime(production['timestamp'])
    quality['timestamp'] = pd.to_datetime(quality['timestamp'])
    maintenance['timestamp'] = pd.to_datetime(maintenance['timestamp'])
    
    return production, quality, maintenance, cnc_predictions, quality_predictions

# Sidebar
st.sidebar.title("Smart Manufacturing")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 Production", "🔍 Quality Control", "🔧 Maintenance", "📈 Reports"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

# Load data
production_df, quality_df, maintenance_df, cnc_pred_df, quality_pred_df = load_all_data()

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
    st.title("Manufacturing Overview Dashboard")
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
    st.subheader("Active Alerts")
    
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
        st.subheader("Daily Production Trend")
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
        st.subheader("Quality Trend")
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
        st.subheader("Machine Utilization")
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
        st.subheader("Production by Product Type")
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
    st.title("Production Analytics")
    
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
    st.subheader("Production Heatmap (Hour x Day)")
    
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
    
    # Machine comparison
    st.subheader("Machine Performance Comparison")
    
    machine_stats = production_filtered.groupby('machine_id').agg({
        'units_produced': 'sum',
        'utilization_percent': 'mean',
        'cycle_time_minutes': 'mean'
    }).reset_index()
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Total Production', 'Avg Utilization', 'Avg Cycle Time')
    )
    
    fig.add_trace(
        go.Bar(x=machine_stats['machine_id'], y=machine_stats['units_produced'], 
               name='Units', marker_color='lightblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=machine_stats['machine_id'], y=machine_stats['utilization_percent'],
               name='Utilization %', marker_color='lightgreen'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=machine_stats['machine_id'], y=machine_stats['cycle_time_minutes'],
               name='Cycle Time', marker_color='lightsalmon'),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE: QUALITY CONTROL
# ==============================================================================
elif page == "🔍 Quality Control":
    st.title("Quality Control Dashboard")
    
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
    st.subheader("Statistical Process Control Chart")
    
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
    
    # Pareto chart of defect types
    st.subheader("Defect Types (Pareto Analysis)")
    
    # Parse defect types
    defect_types_list = []
    for types_str in quality_filtered['defect_types'].dropna():
        if types_str:
            defect_types_list.extend(types_str.split(','))
    
    if defect_types_list:
        defect_counts = pd.Series(defect_types_list).value_counts().reset_index()
        defect_counts.columns = ['defect_type', 'count']
        defect_counts['cumulative_pct'] = defect_counts['count'].cumsum() / defect_counts['count'].sum() * 100
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=defect_counts['defect_type'], y=defect_counts['count'],
                   name='Count', marker_color='skyblue'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=defect_counts['defect_type'], y=defect_counts['cumulative_pct'],
                      name='Cumulative %', line=dict(color='red', width=2), mode='lines+markers'),
            secondary_y=True
        )
        
        fig.update_layout(title='Defect Types - Pareto Chart', height=400)
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE: MAINTENANCE
# ==============================================================================
elif page == "🔧 Maintenance":
    st.title("Predictive Maintenance Dashboard")
    
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
    st.subheader("Machine Health Status")
    
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
    
    st.markdown("---")
    
    # Maintenance timeline
    st.subheader("Maintenance History Timeline")
    
    fig = px.scatter(
        maintenance_df,
        x='timestamp',
        y='machine_id',
        color='maintenance_type',
        size='duration_hours',
        hover_data=['description', 'cost_usd'],
        title='Maintenance Events Over Time'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE: REPORTS
# ==============================================================================
elif page == "📈 Reports":
    st.title("Executive Reports")
    
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
    
    # ROI Calculation
    st.subheader("💰 Cost Savings Analysis")
    
    # Calculate potential savings
    prevented_downtime = 5  # Estimated downtime events prevented
    cost_per_downtime = 5000  # USD
    predicted_savings = prevented_downtime * cost_per_downtime
    
    quality_improvement = 0.001  # 0.1% reduction in defect rate
    units_saved = production_filtered['units_produced'].sum() * quality_improvement
    cost_per_unit = 50  # USD
    quality_savings = units_saved * cost_per_unit
    
    total_savings = predicted_savings + quality_savings
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted Maintenance Savings",
            f"${predicted_savings:,.0f}",
            delta="Prevented downtime"
        )
    
    with col2:
        st.metric(
            "Quality Improvement Savings",
            f"${quality_savings:,.0f}",
            delta=f"{units_saved:.0f} units saved"
        )
    
    with col3:
        st.metric(
            "Total Estimated Savings",
            f"${total_savings:,.0f}",
            delta="per period"
        )
    
    st.markdown("---")
    
    # Download reports
    st.subheader("Export Data")
    
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