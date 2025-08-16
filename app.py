import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
from scipy import stats
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Configure page settings
st.set_page_config(
    page_title="🌍 Climate Data Explorer",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
        padding-left: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ClimateDataApp:
    """
    Main application class for climate data analysis
    
    Professional architecture with separation of concerns:
    - Data management
    - Analysis functions
    - Visualization methods
    - User interface components
    """
    
    def __init__(self):
        """Initialize the application with default settings"""
        self.data = None
        self.processed_data = None
        
        # Professional color schemes for climate visualization
        self.climate_colors = {
            'temperature': ['#313695', '#4575b4', '#74add1', '#abd9e9',
                           '#e0f3f8', '#fee090', '#fdae61', '#f46d43', 
                           '#d73027', '#a50026'],
            'warming_cooling': ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
                               '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        }
        
        # Cache for improved performance
        if 'data_cache' not in st.session_state:
            st.session_state.data_cache = {}
    
    @st.cache_data
    def download_nasa_data(_self):
        """
        Download NASA GISS temperature data with caching for performance
        
        Teaching moment: Always cache expensive operations in web apps
        """
        try:
            url = "https://data.giss.nasa.gov/gistemp/graphs/graph_data/Global_Mean_Estimates_based_on_Land_and_Ocean_Data/graph.txt"
            
            with st.spinner("🛰️ Downloading latest NASA climate data..."):
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Parse NASA data format
                lines = response.text.strip().split('\n')
                years, temps = [], []
                
                for line in lines:
                    if line.strip() and not line.startswith('*') and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                year = int(parts[0])
                                temp = float(parts[1])
                                years.append(year)
                                temps.append(temp)
                            except ValueError:
                                continue
                
                data = pd.DataFrame({
                    'Year': years,
                    'Temperature_Anomaly': temps
                })
                
                st.success(f"✅ Successfully loaded {len(data)} years of NASA data!")
                return data
                
        except Exception as e:
            st.error(f"❌ Error downloading NASA data: {e}")
            st.info("🔄 Loading demonstration data instead...")
            return self._create_demo_data()
    
    def _create_demo_data(self):
        """Create realistic demonstration data if download fails"""
        np.random.seed(42)
        years = range(1880, 2024)
        
        # Realistic temperature trend with noise
        base_trend = np.linspace(-0.2, 1.1, len(years))
        noise = np.random.normal(0, 0.15, len(years))
        cyclical = 0.1 * np.sin(np.array(years) * 2 * np.pi / 7)
        
        temperature_anomaly = base_trend + noise + cyclical
        
        return pd.DataFrame({
            'Year': years,
            'Temperature_Anomaly': temperature_anomaly
        })
    
    def process_data(self, df):
        """
        Process raw data for comprehensive analysis
        
        Teaching moment: Always enhance your data with derived features
        """
        if df is None:
            return None
        
        # Create enhanced dataset
        processed = df.copy()
        
        # Time-based features
        processed['Decade'] = (processed['Year'] // 10) * 10
        processed['Century'] = np.where(processed['Year'] < 1900, '19th Century',
                                      np.where(processed['Year'] < 2000, '20th Century', '21st Century'))
        
        # Moving averages for trend analysis
        processed['MA_5yr'] = processed['Temperature_Anomaly'].rolling(window=5, center=True).mean()
        processed['MA_10yr'] = processed['Temperature_Anomaly'].rolling(window=10, center=True).mean()
        processed['MA_30yr'] = processed['Temperature_Anomaly'].rolling(window=30, center=True).mean()
        
        # Temperature categories
        processed['Temp_Category'] = pd.cut(
            processed['Temperature_Anomaly'],
            bins=[-np.inf, -0.5, -0.25, 0, 0.25, 0.5, np.inf],
            labels=['Very Cold', 'Cold', 'Cool', 'Warm', 'Hot', 'Very Hot']
        )
        
        # Extreme event detection
        std_dev = processed['Temperature_Anomaly'].std()
        mean_temp = processed['Temperature_Anomaly'].mean()
        processed['Is_Extreme'] = abs(processed['Temperature_Anomaly'] - mean_temp) > 2 * std_dev
        
        # Recent trends (last 40 years)
        processed['Is_Recent'] = processed['Year'] >= (processed['Year'].max() - 40)
        
        return processed
    
    def calculate_statistics(self, df):
        """Calculate comprehensive climate statistics"""
        if df is None:
            return {}
        
        # Overall statistics
        stats_dict = {
            'data_range': f"{df['Year'].min()} - {df['Year'].max()}",
            'total_years': len(df),
            'mean_anomaly': df['Temperature_Anomaly'].mean(),
            'std_anomaly': df['Temperature_Anomaly'].std(),
            'max_anomaly': df['Temperature_Anomaly'].max(),
            'min_anomaly': df['Temperature_Anomaly'].min(),
            'max_year': df.loc[df['Temperature_Anomaly'].idxmax(), 'Year'],
            'min_year': df.loc[df['Temperature_Anomaly'].idxmin(), 'Year']
        }
        
        # Trend analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['Year'], df['Temperature_Anomaly']
        )
        
        stats_dict.update({
            'warming_rate': slope * 10,  # per decade
            'correlation': r_value,
            'p_value': p_value,
            'trend_significant': p_value < 0.05
        })
        
        # Recent trends (last 50 years)
        recent_data = df[df['Year'] >= df['Year'].max() - 50]
        if len(recent_data) > 10:
            recent_slope, _, recent_r, recent_p, _ = stats.linregress(
                recent_data['Year'], recent_data['Temperature_Anomaly']
            )
            stats_dict.update({
                'recent_warming_rate': recent_slope * 10,
                'recent_correlation': recent_r,
                'recent_p_value': recent_p
            })
        
        # Extreme events
        stats_dict['extreme_years'] = df['Is_Extreme'].sum()
        stats_dict['warming_years'] = (df['Temperature_Anomaly'] > 0).sum()
        stats_dict['cooling_years'] = (df['Temperature_Anomaly'] < 0).sum()
        
        return stats_dict
    
    def create_interactive_timeseries(self, df):
        """Create interactive time series plot with Plotly"""
        fig = go.Figure()
        
        # Add main temperature line
        fig.add_trace(go.Scatter(
            x=df['Year'],
            y=df['Temperature_Anomaly'],
            mode='lines',
            name='Annual Anomaly',
            line=dict(color='lightblue', width=1),
            hovertemplate='<b>Year:</b> %{x}<br><b>Anomaly:</b> %{y:.2f}°C<extra></extra>'
        ))
        
        # Add 10-year moving average
        fig.add_trace(go.Scatter(
            x=df['Year'],
            y=df['MA_10yr'],
            mode='lines',
            name='10-Year Average',
            line=dict(color='red', width=3),
            hovertemplate='<b>Year:</b> %{x}<br><b>10-yr Avg:</b> %{y:.2f}°C<extra></extra>'
        ))
        
        # Add zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        # Highlight extreme years
        extreme_data = df[df['Is_Extreme'] == True]
        if not extreme_data.empty:
            fig.add_trace(go.Scatter(
                x=extreme_data['Year'],
                y=extreme_data['Temperature_Anomaly'],
                mode='markers',
                name='Extreme Years',
                marker=dict(color='red', size=8, symbol='diamond'),
                hovertemplate='<b>Extreme Year:</b> %{x}<br><b>Anomaly:</b> %{y:.2f}°C<extra></extra>'
            ))
        
        # Customize layout
        fig.update_layout(
            title={
                'text': 'Global Temperature Anomalies (1880-Present)',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='Year',
            yaxis_title='Temperature Anomaly (°C)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_distribution_plot(self, df):
        """Create temperature distribution analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature Distribution', 'Decade Comparison', 
                           'Warming vs Cooling', 'Recent Trends'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # 1. Temperature distribution histogram
        fig.add_trace(
            go.Histogram(x=df['Temperature_Anomaly'], nbinsx=30, name='Distribution',
                        marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. Decade comparison
        decade_avg = df.groupby('Decade')['Temperature_Anomaly'].mean().reset_index()
        fig.add_trace(
            go.Bar(x=decade_avg['Decade'], y=decade_avg['Temperature_Anomaly'],
                  name='Decade Average', marker_color='orange'),
            row=1, col=2
        )
        
        # 3. Warming vs cooling pie chart
        warming_count = (df['Temperature_Anomaly'] > 0).sum()
        cooling_count = (df['Temperature_Anomaly'] <= 0).sum()
        
        fig.add_trace(
            go.Pie(labels=['Warming Years', 'Cooling Years'],
                  values=[warming_count, cooling_count],
                  marker_colors=['red', 'blue']),
            row=2, col=1
        )
        
        # 4. Recent trends scatter
        recent_data = df[df['Year'] >= 1980]
        fig.add_trace(
            go.Scatter(x=recent_data['Year'], y=recent_data['Temperature_Anomaly'],
                      mode='markers+lines', name='Recent Trends',
                      marker_color='green'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Climate Data Analysis Dashboard")
        
        return fig
    
    def create_heatmap(self, df):
        """Create decade-year heatmap visualization"""
        # Prepare data for heatmap
        df_heatmap = df.copy()
        df_heatmap['Decade_Start'] = (df_heatmap['Year'] // 10) * 10
        df_heatmap['Year_in_Decade'] = df_heatmap['Year'] % 10
        
        # Create pivot table
        heatmap_data = df_heatmap.pivot_table(
            values='Temperature_Anomaly',
            index='Decade_Start',
            columns='Year_in_Decade',
            aggfunc='mean'
        )
        
        # Create heatmap with Plotly
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[f'Year {i}' for i in range(10)],
            y=[f'{int(decade)}s' for decade in heatmap_data.index],
            colorscale='RdYlBu_r',
            zmid=0,
            hovertemplate='<b>Decade:</b> %{y}<br><b>Year in Decade:</b> %{x}<br><b>Anomaly:</b> %{z:.2f}°C<extra></extra>'
        ))
        
        fig.update_layout(
            title='Temperature Anomaly Heatmap by Decade',
            xaxis_title='Year in Decade',
            yaxis_title='Decade',
            height=400
        )
        
        return fig
    
    def generate_report(self, stats):
        """Generate automated climate analysis report"""
        warming_rate = stats.get('warming_rate', 0)
        recent_rate = stats.get('recent_warming_rate', 0)
        
        report = f"""
        ## 📊 Climate Analysis Report
        
        **Data Overview:**
        - 📅 Time Period: {stats.get('data_range', 'N/A')}
        - 📈 Total Years: {stats.get('total_years', 0)}
        - 🌡️ Average Anomaly: {stats.get('mean_anomaly', 0):.3f}°C
        
        **Key Findings:**
        - 🔥 **Warming Trend**: {warming_rate:.4f}°C per decade
        - 📊 **Statistical Confidence**: R = {stats.get('correlation', 0):.3f}
        - ✅ **Significance**: {'Highly significant' if stats.get('trend_significant', False) else 'Not significant'}
        
        **Temperature Extremes:**
        - 🥵 **Warmest Year**: {stats.get('max_year', 'N/A')} ({stats.get('max_anomaly', 0):.2f}°C)
        - 🥶 **Coolest Year**: {stats.get('min_year', 'N/A')} ({stats.get('min_anomaly', 0):.2f}°C)
        - ⚡ **Extreme Events**: {stats.get('extreme_years', 0)} years
        
        **Recent Trends (Last 50 Years):**
        - 🚀 **Acceleration**: {recent_rate:.4f}°C per decade
        - 📈 **Warming Years**: {stats.get('warming_years', 0)} out of {stats.get('total_years', 0)}
        
        **Climate Context:**
        """
        
        if warming_rate > 0.1:
            report += "\n- ⚠️ **Significant warming trend detected** - consistent with global climate change"
        if recent_rate > warming_rate * 1.5:
            report += "\n- 🚨 **Accelerating warming** in recent decades"
        if stats.get('extreme_years', 0) > stats.get('total_years', 1) * 0.05:
            report += "\n- 🌡️ **High frequency of extreme temperature events**"
        
        return report

def main():
    """Main application function"""
    
    # Initialize the app
    app = ClimateDataApp()
    
    # Header
    st.markdown('<div class="main-header">🌍 Climate Data Explorer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #7f8c8d;">
            Professional climate data analysis and visualization platform<br>
            <em>Powered by NASA GISS temperature data</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Analysis Controls")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Select Data Source:",
        ["NASA GISS (Live Data)", "Demonstration Data"],
        help="Choose between live NASA data or demonstration dataset"
    )
    
    # Load data based on selection
    if data_source == "NASA GISS (Live Data)":
        if st.sidebar.button("🔄 Load NASA Data", type="primary"):
            app.data = app.download_nasa_data()
    else:
        app.data = app._create_demo_data()
        st.sidebar.info("Using demonstration data for analysis")
    
    # Process data if available
    if app.data is not None:
        app.processed_data = app.process_data(app.data)
        
        # Time period filter
        st.sidebar.markdown("### 📅 Time Period Filter")
        year_range = st.sidebar.slider(
            "Select Year Range:",
            min_value=int(app.data['Year'].min()),
            max_value=int(app.data['Year'].max()),
            value=(int(app.data['Year'].min()), int(app.data['Year'].max())),
            step=1
        )
        
        # Filter data based on year range
        filtered_data = app.processed_data[
            (app.processed_data['Year'] >= year_range[0]) & 
            (app.processed_data['Year'] <= year_range[1])
        ]
        
        # Calculate statistics
        stats = app.calculate_statistics(filtered_data)
        
        # Display key metrics
        st.markdown('<div class="sub-header">📊 Key Climate Metrics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🌡️ Warming Rate",
                f"{stats.get('warming_rate', 0):.4f}°C/decade",
                delta=f"R²={stats.get('correlation', 0)**2:.3f}"
            )
        
        with col2:
            st.metric(
                "🥵 Warmest Year",
                f"{stats.get('max_year', 'N/A')}",
                delta=f"{stats.get('max_anomaly', 0):.2f}°C"
            )
        
        with col3:
            st.metric(
                "📈 Data Points",
                f"{stats.get('total_years', 0)} years",
                delta=f"{stats.get('extreme_years', 0)} extreme"
            )
        
        with col4:
            warming_pct = (stats.get('warming_years', 0) / stats.get('total_years', 1)) * 100
            st.metric(
                "🔥 Warming Years",
                f"{warming_pct:.1f}%",
                delta=f"{stats.get('warming_years', 0)}/{stats.get('total_years', 0)}"
            )
        
        # Main visualizations
        st.markdown('<div class="sub-header">📈 Interactive Visualizations</div>', unsafe_allow_html=True)
        
        # Tab layout for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌡️ Temperature Trends", 
            "📊 Statistical Analysis", 
            "🔥 Heatmap View", 
            "📋 Analysis Report"
        ])
        
        with tab1:
            st.plotly_chart(
                app.create_interactive_timeseries(filtered_data), 
                use_container_width=True
            )
            
            st.markdown("""
            **💡 How to Read This Chart:**
            - **Blue line**: Annual temperature anomalies (difference from 20th century average)
            - **Red line**: 10-year moving average (smoothed trend)
            - **Red diamonds**: Extreme temperature years (>2 standard deviations)
            - **Hover** over points for detailed information
            """)
        
        with tab2:
            st.plotly_chart(
                app.create_distribution_plot(filtered_data), 
                use_container_width=True
            )
            
            st.markdown("""
            **📊 Statistical Insights:**
            - **Top Left**: Distribution shows shift toward warming
            - **Top Right**: Clear acceleration in recent decades  
            - **Bottom Left**: Proportion of warming vs cooling years
            - **Bottom Right**: Recent trend detail (1980+)
            """)
        
        with tab3:
            st.plotly_chart(
                app.create_heatmap(filtered_data), 
                use_container_width=True
            )
            
            st.markdown("""
            **🔥 Heatmap Analysis:**
            - **Rows**: Each decade from 1880s to present
            - **Columns**: Years 0-9 within each decade
            - **Colors**: Red = warming, Blue = cooling
            - **Pattern**: Notice increasing red in recent decades
            """)
        
        with tab4:
            st.markdown(app.generate_report(stats))
            
            # Download options
            st.markdown("### 💾 Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download processed data
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="📁 Download Data (CSV)",
                    data=csv,
                    file_name=f"climate_data_{year_range[0]}_{year_range[1]}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download analysis report
                report_text = app.generate_report(stats)
                st.download_button(
                    label="📄 Download Report (TXT)",
                    data=report_text,
                    file_name=f"climate_report_{year_range[0]}_{year_range[1]}.txt",
                    mime="text/plain"
                )
    
    else:
        # No data loaded state
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Getting Started</h3>
            <p>Click <strong>"Load NASA Data"</strong> in the sidebar to begin your climate analysis journey!</p>
            <p>This app will download the latest temperature data from NASA GISS and create professional visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        st.markdown("### ✨ App Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🌍 Real NASA Data**
            - Live GISS temperature data
            - Global temperature anomalies
            - Historical records since 1880
            """)
        
        with col2:
            st.markdown("""
            **📊 Professional Analysis**
            - Statistical trend analysis
            - Moving averages & smoothing
            - Extreme event detection
            """)
        
        with col3:
            st.markdown("""
            **🎨 Interactive Visualizations**
            - Plotly interactive charts
            - Multiple view perspectives
            - Downloadable reports
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; margin-top: 2rem;">
        <p>🌍 <strong>Climate Data Explorer</strong> | Built with ❤️ for climate science education</p>
        <p><em>Data Source: NASA Goddard Institute for Space Studies (GISS)</em></p>
        <p>Professional climate analysis tools for researchers, educators, and climate enthusiasts</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
