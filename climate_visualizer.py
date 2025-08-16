import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import asyncio
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from main import MentalHealthAnalyzerSystem, MentalHealthSeverity
import time
import base64
import io

# Configure page
st.set_page_config(
    page_title="Mental Health AI Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ff4b4b;
        padding: 1rem;
        border-radius: 5px;
        color: white;
        margin: 1rem 0;
    }
    .alert-moderate {
        background-color: #ffa500;
        padding: 1rem;
        border-radius: 5px;
        color: white;
        margin: 1rem 0;
    }
    .insight-box {
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    data = []
    sample_texts = [
        "I've been feeling really down lately, nothing seems to bring me joy",
        "Anxiety is overwhelming me, can't stop worrying about everything",
        "Work stress is killing me, I feel completely burnt out",
        "Having trouble sleeping, mind racing with negative thoughts",
        "Feeling hopeless about the future, nothing seems worth it",
        "Great day today! Feeling positive and energetic",
        "Managed my stress well with meditation and exercise",
        "Therapy session was really helpful, making progress",
        "Spending time with friends lifted my mood significantly",
        "Accomplished a lot today, feeling proud of myself"
    ]
    
    for i, date in enumerate(dates[:200]):  # Last 200 days
        text = sample_texts[i % len(sample_texts)]
        # Simulate varying scores based on text sentiment
        if any(word in text.lower() for word in ['down', 'anxiety', 'stress', 'hopeless']):
            depression = np.random.uniform(0.6, 0.9)
            anxiety = np.random.uniform(0.5, 0.8)
            stress = np.random.uniform(0.4, 0.9)
        else:
            depression = np.random.uniform(0.1, 0.4)
            anxiety = np.random.uniform(0.1, 0.3)
            stress = np.random.uniform(0.1, 0.4)
            
        avg_score = (depression + anxiety + stress) / 3
        if avg_score >= 0.7:
            severity = 'critical'
        elif avg_score >= 0.5:
            severity = 'high'
        elif avg_score >= 0.3:
            severity = 'moderate'
        else:
            severity = 'low'
            
        data.append({
            'date': date,
            'text': text,
            'depression_score': depression,
            'anxiety_score': anxiety,
            'stress_score': stress,
            'severity': severity,
            'confidence': np.random.uniform(0.7, 0.95),
            'user_id': f'user_{np.random.randint(1, 50)}'
        })
    
    return pd.DataFrame(data)

def create_severity_gauge(value, title):
    """Create a gauge chart for severity levels"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.6], 'color': "yellow"},
                {'range': [0.6, 0.8], 'color': "orange"},
                {'range': [0.8, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_heatmap_analysis(df):
    """Create correlation heatmap"""
    corr_data = df[['depression_score', 'anxiety_score', 'stress_score', 'confidence']].corr()
    
    fig = ff.create_annotated_heatmap(
        z=corr_data.values,
        x=list(corr_data.columns),
        y=list(corr_data.index),
        annotation_text=corr_data.round(2).values,
        showscale=True,
        colorscale='RdBu'
    )
    fig.update_layout(title='Mental Health Metrics Correlation Analysis')
    return fig

def create_time_series_analysis(df):
    """Create advanced time series analysis"""
    # Resample data by week
    df_weekly = df.set_index('date').resample('W').mean()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Depression Trends', 'Anxiety Trends', 'Stress Trends', 'Overall Risk Level'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]]
    )
    
    # Depression trend with confidence intervals
    fig.add_trace(
        go.Scatter(
            x=df_weekly.index, 
            y=df_weekly['depression_score'],
            mode='lines+markers',
            name='Depression',
            line=dict(color='blue', width=3)
        ),
        row=1, col=1
    )
    
    # Anxiety trend
    fig.add_trace(
        go.Scatter(
            x=df_weekly.index, 
            y=df_weekly['anxiety_score'],
            mode='lines+markers',
            name='Anxiety',
            line=dict(color='orange', width=3)
        ),
        row=1, col=2
    )
    
    # Stress trend
    fig.add_trace(
        go.Scatter(
            x=df_weekly.index, 
            y=df_weekly['stress_score'],
            mode='lines+markers',
            name='Stress',
            line=dict(color='red', width=3)
        ),
        row=2, col=1
    )
    
    # Overall risk (average of all scores)
    df_weekly['overall_risk'] = (df_weekly['depression_score'] + 
                                df_weekly['anxiety_score'] + 
                                df_weekly['stress_score']) / 3
    
    fig.add_trace(
        go.Scatter(
            x=df_weekly.index, 
            y=df_weekly['overall_risk'],
            mode='lines+markers',
            name='Overall Risk',
            fill='tonexty',
            line=dict(color='purple', width=3)
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="Advanced Time Series Analysis")
    return fig

def create_user_segmentation(df):
    """Create user segmentation analysis"""
    user_analysis = df.groupby('user_id').agg({
        'depression_score': 'mean',
        'anxiety_score': 'mean', 
        'stress_score': 'mean',
        'confidence': 'mean'
    }).reset_index()
    
    user_analysis['risk_level'] = (user_analysis['depression_score'] + 
                                  user_analysis['anxiety_score'] + 
                                  user_analysis['stress_score']) / 3
    
    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=user_analysis['depression_score'],
        y=user_analysis['anxiety_score'],
        z=user_analysis['stress_score'],
        mode='markers',
        marker=dict(
            size=user_analysis['confidence'] * 20,
            color=user_analysis['risk_level'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Risk Level")
        ),
        text=user_analysis['user_id'],
        hovertemplate='<b>%{text}</b><br>' +
                      'Depression: %{x:.2f}<br>' +
                      'Anxiety: %{y:.2f}<br>' +
                      'Stress: %{z:.2f}<br>' +
                      '<extra></extra>'
    )])
    
    fig.update_layout(
        title='3D User Risk Segmentation',
        scene=dict(
            xaxis_title='Depression Score',
            yaxis_title='Anxiety Score',
            zaxis_title='Stress Score'
        ),
        height=600
    )
    return fig

def create_wordcloud_analysis(df):
    """Create word cloud from text data"""
    all_text = ' '.join(df['text'].values)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(all_text)
    
    # Convert to image for streamlit
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Mental Health AI Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Advanced NLP-powered Mental Health Monitoring System")
    
    # Load data
    df = load_sample_data()
    
    # Sidebar controls
    st.sidebar.header("📊 Dashboard Controls")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Analysis Period",
        value=(df['date'].min().date(), df['date'].max().date()),
        min_value=df['date'].min().date(),
        max_value=df['date'].max().date()
    )
    
    # Severity filter
    severity_options = ['low', 'moderate', 'high', 'critical']
    selected_severities = st.sidebar.multiselect(
        "Filter by Severity Level",
        severity_options,
        default=severity_options
    )
    
    # User filter
    user_options = df['user_id'].unique()
    selected_users = st.sidebar.multiselect(
        "Filter by Users",
        user_options,
        default=user_options[:10]  # Default to first 10 users
    )
    
    # Apply filters
    mask = (
        (df['date'].dt.date >= date_range[0]) & 
        (df['date'].dt.date <= date_range[1]) &
        (df['severity'].isin(selected_severities)) &
        (df['user_id'].isin(selected_users))
    )
    filtered_df = df[mask]
    
    # Real-time simulation toggle
    real_time = st.sidebar.checkbox("Enable Real-time Simulation", value=False)
    
    if real_time:
        # Auto-refresh every 5 seconds
        time.sleep(1)
        st.rerun()
    
    # Key Metrics Row
    st.markdown("## 📈 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_analyses = len(filtered_df)
        st.metric("Total Analyses", total_analyses, delta=f"+{np.random.randint(5, 15)}")
    
    with col2:
        avg_depression = filtered_df['depression_score'].mean()
        prev_depression = avg_depression + np.random.uniform(-0.1, 0.1)
        st.metric("Avg Depression", f"{avg_depression:.2f}", 
                 delta=f"{avg_depression - prev_depression:.2f}")
    
    with col3:
        avg_anxiety = filtered_df['anxiety_score'].mean()
        prev_anxiety = avg_anxiety + np.random.uniform(-0.1, 0.1)
        st.metric("Avg Anxiety", f"{avg_anxiety:.2f}", 
                 delta=f"{avg_anxiety - prev_anxiety:.2f}")
    
    with col4:
        high_risk_count = len(filtered_df[filtered_df['severity'].isin(['high', 'critical'])])
        st.metric("High Risk Cases", high_risk_count, 
                 delta=f"+{np.random.randint(0, 5)}")
    
    with col5:
        avg_confidence = filtered_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}",
                 delta=f"+{np.random.uniform(0, 0.05):.2f}")
    
    # Alert System
    critical_cases = len(filtered_df[filtered_df['severity'] == 'critical'])
    if critical_cases > 0:
        st.markdown(f'<div class="alert-high">🚨 ALERT: {critical_cases} critical cases detected requiring immediate attention!</div>', 
                   unsafe_allow_html=True)
    
    high_cases = len(filtered_df[filtered_df['severity'] == 'high'])
    if high_cases > 5:
        st.markdown(f'<div class="alert-moderate">⚠️ WARNING: {high_cases} high-risk cases detected</div>', 
                   unsafe_allow_html=True)
    
    # Main Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "📈 Time Series", "👥 User Analysis", 
        "🔍 Text Analysis", "🎯 Predictions", "⚙️ Model Performance"
    ])
    
    with tab1:
        st.header("Dashboard Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity distribution
            severity_counts = filtered_df['severity'].value_counts()
            fig_severity = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Mental Health Severity Distribution",
                color_discrete_map={
                    'low': '#90EE90',
                    'moderate': '#FFD700', 
                    'high': '#FF8C00',
                    'critical': '#FF4500'
                }
            )
            fig_severity.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_severity, use_container_width=True)
        
        with col2:
            # Gauge charts for current levels
            current_depression = filtered_df['depression_score'].iloc[-1] if len(filtered_df) > 0 else 0.5
            fig_gauge = create_severity_gauge(current_depression, "Current Depression Level")
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Mental Health Metrics Correlation")
        fig_heatmap = create_heatmap_analysis(filtered_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        st.header("Time Series Analysis")
        
        # Advanced time series
        fig_timeseries = create_time_series_analysis(filtered_df)
        st.plotly_chart(fig_timeseries, use_container_width=True)
        
        # Daily patterns
        st.subheader("Daily Patterns Analysis")
        filtered_df['hour'] = filtered_df['date'].dt.hour
        filtered_df['day_of_week'] = filtered_df['date'].dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            hourly_risk = filtered_df.groupby('hour')[['depression_score', 'anxiety_score', 'stress_score']].mean()
            fig_hourly = px.line(
                hourly_risk.reset_index(), 
                x='hour', 
                y=['depression_score', 'anxiety_score', 'stress_score'],
                title="Risk Levels by Hour of Day"
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            daily_risk = filtered_df.groupby('day_of_week')[['depression_score', 'anxiety_score', 'stress_score']].mean()
            fig_daily = px.bar(
                daily_risk.reset_index(), 
                x='day_of_week', 
                y=['depression_score', 'anxiety_score', 'stress_score'],
                title="Average Risk by Day of Week",
                barmode='group'
            )
            st.plotly_chart(fig_daily, use_container_width=True)
    
    with tab3:
        st.header("User Segmentation & Risk Analysis")
        
        # 3D user segmentation
        fig_3d = create_user_segmentation(filtered_df)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # User risk ranking
        st.subheader("High-Risk User Identification")
        user_risk = filtered_df.groupby('user_id').agg({
            'depression_score': 'mean',
            'anxiety_score': 'mean',
            'stress_score': 'mean',
            'confidence': 'mean'
        }).reset_index()
        
        user_risk['overall_risk'] = (user_risk['depression_score'] + 
                                   user_risk['anxiety_score'] + 
                                   user_risk['stress_score']) / 3
        
        user_risk = user_risk.sort_values('overall_risk', ascending=False)
        
        # Format the dataframe for display
        display_df = user_risk.head(10).copy()
        for col in ['depression_score', 'anxiety_score', 'stress_score', 'overall_risk', 'confidence']:
            display_df[col] = display_df[col].round(3)
        
        st.dataframe(
            display_df,
            column_config={
                "user_id": "User ID",
                "depression_score": st.column_config.ProgressColumn(
                    "Depression",
                    help="Depression risk score",
                    min_value=0,
                    max_value=1,
                ),
                "anxiety_score": st.column_config.ProgressColumn(
                    "Anxiety", 
                    help="Anxiety risk score",
                    min_value=0,
                    max_value=1,
                ),
                "stress_score": st.column_config.ProgressColumn(
                    "Stress",
                    help="Stress risk score", 
                    min_value=0,
                    max_value=1,
                ),
                "overall_risk": st.column_config.ProgressColumn(
                    "Overall Risk",
                    help="Combined risk score",
                    min_value=0,
                    max_value=1,
                ),
            },
            hide_index=True,
            use_container_width=True
        )
    
    with tab4:
        st.header("Text Analysis & Insights")
        
        # Word cloud
        st.subheader("Most Common Terms")
        fig_wordcloud = create_wordcloud_analysis(filtered_df)
        st.pyplot(fig_wordcloud)
        
        # Sentiment distribution
        st.subheader("Sentiment Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Text length analysis
            filtered_df['text_length'] = filtered_df['text'].str.len()
            fig_length = px.histogram(
                filtered_df, 
                x='text_length', 
                color='severity',
                title="Text Length Distribution by Severity"
            )
            st.plotly_chart(fig_length, use_container_width=True)
        
        with col2:
            # Most concerning texts
            st.subheader("Highest Risk Texts")
            high_risk_texts = filtered_df.nlargest(5, 'depression_score')[['text', 'depression_score', 'severity']]
            for idx, row in high_risk_texts.iterrows():
                st.markdown(f"**Risk Score: {row['depression_score']:.3f}** ({row['severity']})")
                st.markdown(f"_{row['text'][:100]}..._")
                st.markdown("---")
    
    with tab5:
        st.header("Live Prediction Interface")
        
        # Text input for real-time analysis
        st.subheader("Analyze New Text")
        user_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste text here for mental health analysis...",
            height=100
        )
        
        if st.button("Analyze Text", type="primary"):
            if user_input:
                # Initialize system and get prediction
                system = MentalHealthAnalyzerSystem()
                prediction = system.analyze_single_text(user_input)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Depression Risk", f"{prediction.depression_score:.3f}")
                with col2:
                    st.metric("Anxiety Risk", f"{prediction.anxiety_score:.3f}")
                with col3:
                    st.metric("Stress Risk", f"{prediction.stress_score:.3f}")
                
                # Severity and recommendations
                severity_color = {
                    'low': '🟢', 'moderate': '🟡', 'high': '🟠', 'critical': '🔴'
                }
                
                st.markdown(f"### {severity_color.get(prediction.overall_severity.value, '⚪')} Severity: {prediction.overall_severity.value.title()}")
                st.markdown(f"**Confidence:** {prediction.confidence:.1%}")
                
                # Recommendations
                st.subheader("🎯 Recommendations")
                for i, rec in enumerate(prediction.recommendations, 1):
                    st.markdown(f"{i}. {rec}")
            else:
                st.warning("Please enter some text to analyze.")
    
    with tab6:
        st.header("Model Performance Metrics")
        
        # Simulate model performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Accuracy", "94.2%", delta="2.1%")
        with col2:
            st.metric("Precision", "91.8%", delta="1.5%")
        with col3:
            st.metric("Recall", "89.3%", delta="-0.8%")
        
        # Performance over time
        performance_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
        performance_data = pd.DataFrame({
            'date': performance_dates,
            'accuracy': np.random.uniform(0.85, 0.95, len(performance_dates)),
            'precision': np.random.uniform(0.82, 0.93, len(performance_dates)),
            'recall': np.random.uniform(0.80, 0.91, len(performance_dates))
        })
        
        fig_performance = px.line(
            performance_data, 
            x='date', 
            y=['accuracy', 'precision', 'recall'],
            title="Model Performance Over Time",
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance Analysis")
        features = ['Sentiment Score', 'Text Length', 'Keyword Count', 'Emotional Tone', 
                   'Linguistic Complexity', 'Previous History', 'Time of Day', 'Social Context']
        importance = np.random.uniform(0.05, 0.25, len(features))
        importance = importance / importance.sum()  # Normalize
        
        fig_importance = px.bar(
            x=importance, 
            y=features,
            orientation='h',
            title="Feature Importance in Mental Health Prediction"
        )
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🧠 AI Mental Health Analyzer | Built with Advanced NLP & Machine Learning<br>
        <small>For demonstration purposes only. Not a substitute for professional medical advice.</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
