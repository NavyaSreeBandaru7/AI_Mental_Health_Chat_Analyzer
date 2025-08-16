import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
import warnings
from scipy import stats
from matplotlib.patches import Rectangle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ClimateDataVisualizer:
    """
    A comprehensive class for downloading, processing, and visualizing climate data
    """
    
    def __init__(self):
        """Initialize the visualizer with default settings"""
        self.data = None
        self.processed_data = None
        
        # Set up plotting style - this is crucial for professional visualizations
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Professional color schemes for different types of visualizations
        self.temp_colors = ['#08519c', '#3182bd', '#6baed6', '#bdd7e7', 
                           '#eff3ff', '#fee5d9', '#fcae91', '#fb6a4a', 
                           '#de2d26', '#a50f15']
        
    def download_nasa_data(self, save_path='nasa_temp_data.csv'):
        """
        Download NASA GISS Surface Temperature Analysis data
        
        Teaching moment: Always handle data download gracefully with error checking
        """
        print("📡 Downloading NASA GISS Surface Temperature Data...")
        
        # NASA GISS Global Temperature Anomaly Data URL
        # This is the official NASA dataset - always use authoritative sources!
        url = "https://data.giss.nasa.gov/gistemp/graphs/graph_data/Global_Mean_Estimates_based_on_Land_and_Ocean_Data/graph.txt"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the NASA data format (it's space-separated with headers)
            lines = response.text.strip().split('\n')
            
            # Find where the actual data starts (skip header comments)
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('*') and not line.startswith('#'):
                    data_start = i
                    break
            
            # Parse the data into a structured format
            years = []
            temps = []
            
            for line in lines[data_start:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            year = int(parts[0])
                            temp = float(parts[1])
                            years.append(year)
                            temps.append(temp)
                        except (ValueError, IndexError):
                            continue
            
            # Create DataFrame
            self.data = pd.DataFrame({
                'Year': years,
                'Temperature_Anomaly': temps
            })
            
            # Save for future use
            self.data.to_csv(save_path, index=False)
            print(f"✅ Data downloaded successfully! {len(self.data)} records saved to {save_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            print("💡 Tip: Check your internet connection or try again later")
            return False
    
    def load_sample_data(self):
        """
        Create sample data if download fails - for learning purposes
        """
        print("📊 Creating sample temperature data for demonstration...")
        
        # Generate realistic temperature anomaly data based on actual trends
        np.random.seed(42)  # For reproducible results
        years = range(1880, 2024)
        
        # Create a realistic trend with noise
        base_trend = np.linspace(-0.2, 1.1, len(years))  # Overall warming trend
        noise = np.random.normal(0, 0.15, len(years))
        
        # Add some cyclical patterns (like El Niño/La Niña)
        cyclical = 0.1 * np.sin(np.array(years) * 2 * np.pi / 7) + 0.05 * np.sin(np.array(years) * 2 * np.pi / 3)
        
        temperature_anomaly = base_trend + noise + cyclical
        
        self.data = pd.DataFrame({
            'Year': years,
            'Temperature_Anomaly': temperature_anomaly
        })
        
        print(f"✅ Sample data created with {len(self.data)} records")
    
    def process_data(self):
        """
        Process and enhance the temperature data for analysis
        
        Teaching moment: Data preprocessing is crucial for meaningful analysis
        """
        if self.data is None:
            print("❌ No data available. Please download or load data first.")
            return
        
        print("🔄 Processing temperature data...")
        
        # Create a copy for processing
        df = self.data.copy()
        
        # Add time-based features for analysis
        df['Decade'] = (df['Year'] // 10) * 10
        df['Century'] = np.where(df['Year'] < 1900, '19th Century',
                                np.where(df['Year'] < 2000, '20th Century', '21st Century'))
        
        # Calculate moving averages for trend analysis
        df['MA_5yr'] = df['Temperature_Anomaly'].rolling(window=5, center=True).mean()
        df['MA_10yr'] = df['Temperature_Anomaly'].rolling(window=10, center=True).mean()
        df['MA_30yr'] = df['Temperature_Anomaly'].rolling(window=30, center=True).mean()
        
        # Calculate temperature categories
        df['Temp_Category'] = pd.cut(df['Temperature_Anomaly'], 
                                   bins=[-np.inf, -0.5, -0.25, 0, 0.25, 0.5, np.inf],
                                   labels=['Very Cold', 'Cold', 'Cool', 'Warm', 'Hot', 'Very Hot'])
        
        # Identify significant changes (anomalies beyond 2 standard deviations)
        std_dev = df['Temperature_Anomaly'].std()
        mean_temp = df['Temperature_Anomaly'].mean()
        df['Is_Extreme'] = abs(df['Temperature_Anomaly'] - mean_temp) > 2 * std_dev
        
        self.processed_data = df
        print("✅ Data processing complete!")
        
        # Print some interesting statistics
        print(f"\n📊 Quick Statistics:")
        print(f"   • Data range: {df['Year'].min()} - {df['Year'].max()}")
        print(f"   • Average temperature anomaly: {mean_temp:.3f}°C")
        print(f"   • Highest anomaly: {df['Temperature_Anomaly'].max():.3f}°C in {df.loc[df['Temperature_Anomaly'].idxmax(), 'Year']}")
        print(f"   • Lowest anomaly: {df['Temperature_Anomaly'].min():.3f}°C in {df.loc[df['Temperature_Anomaly'].idxmin(), 'Year']}")
        print(f"   • Extreme years: {df['Is_Extreme'].sum()}")
    
    def create_trend_visualization(self):
        """
        Create the main temperature trend visualization
        
        Teaching moment: Layer multiple visualizations to tell a complete story
        """
        if self.processed_data is None:
            print("❌ Please process data first using process_data()")
            return
        
        print("📈 Creating temperature trend visualization...")
        
        # Create a comprehensive figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Earth\'s Surface Temperature Trends: A Comprehensive Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        df = self.processed_data
        
        # 1. Main time series plot (top left)
        ax1 = axes[0, 0]
        
        # Plot the main temperature anomaly line
        ax1.plot(df['Year'], df['Temperature_Anomaly'], 
                color='lightblue', alpha=0.7, linewidth=0.8, label='Annual Anomaly')
        
        # Add moving averages for trend clarity
        ax1.plot(df['Year'], df['MA_10yr'], 
                color='blue', linewidth=2, label='10-year Average')
        ax1.plot(df['Year'], df['MA_30yr'], 
                color='red', linewidth=2.5, label='30-year Average')
        
        # Add zero reference line
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Highlight extreme years
        extreme_years = df[df['Is_Extreme']]
        ax1.scatter(extreme_years['Year'], extreme_years['Temperature_Anomaly'], 
                   color='red', s=30, alpha=0.8, zorder=5, label='Extreme Years')
        
        ax1.set_title('Global Temperature Anomalies Over Time')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Temperature Anomaly (°C)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Decade comparison (top right)
        ax2 = axes[0, 1]
        decade_avg = df.groupby('Decade')['Temperature_Anomaly'].mean().reset_index()
        
        bars = ax2.bar(decade_avg['Decade'], decade_avg['Temperature_Anomaly'], 
                      color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(decade_avg))))
        
        ax2.set_title('Average Temperature Anomaly by Decade')
        ax2.set_xlabel('Decade')
        ax2.set_ylabel('Average Temperature Anomaly (°C)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, decade_avg['Temperature_Anomaly']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Distribution analysis (bottom left)
        ax3 = axes[1, 0]
        
        # Create histogram with different colors for positive/negative anomalies
        positive_temps = df[df['Temperature_Anomaly'] >= 0]['Temperature_Anomaly']
        negative_temps = df[df['Temperature_Anomaly'] < 0]['Temperature_Anomaly']
        
        ax3.hist(negative_temps, bins=20, alpha=0.7, color='blue', label='Cooling')
        ax3.hist(positive_temps, bins=20, alpha=0.7, color='red', label='Warming')
        
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Distribution of Temperature Anomalies')
        ax3.set_xlabel('Temperature Anomaly (°C)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. Recent trends focus (bottom right)
        ax4 = axes[1, 1]
        recent_data = df[df['Year'] >= 1980]  # Focus on recent decades
        
        # Create a more detailed recent trend
        ax4.fill_between(recent_data['Year'], 0, recent_data['Temperature_Anomaly'],
                        where=(recent_data['Temperature_Anomaly'] >= 0), 
                        color='red', alpha=0.3, label='Above Average')
        ax4.fill_between(recent_data['Year'], 0, recent_data['Temperature_Anomaly'],
                        where=(recent_data['Temperature_Anomaly'] < 0), 
                        color='blue', alpha=0.3, label='Below Average')
        
        ax4.plot(recent_data['Year'], recent_data['Temperature_Anomaly'], 
                color='black', linewidth=1)
        ax4.plot(recent_data['Year'], recent_data['MA_10yr'], 
                color='darkred', linewidth=3, label='10-yr Average')
        
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Recent Temperature Trends (1980-Present)')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Temperature Anomaly (°C)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temperature_trends_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comprehensive visualization created and saved!")
    
    def create_heatmap_visualization(self):
        """
        Create a heatmap showing temperature patterns over time
        
        Teaching moment: Heatmaps are excellent for showing patterns across two dimensions
        """
        if self.processed_data is None:
            print("❌ Please process data first using process_data()")
            return
        
        print("🔥 Creating temperature heatmap...")
        
        df = self.processed_data
        
        # Create decade vs year-in-decade heatmap
        df['Year_in_Decade'] = df['Year'] % 10
        heatmap_data = df.pivot_table(values='Temperature_Anomaly', 
                                     index='Decade', 
                                     columns='Year_in_Decade', 
                                     aggfunc='mean')
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='RdYlBu_r', 
                   center=0,
                   cbar_kws={'label': 'Temperature Anomaly (°C)'})
        
        plt.title('Temperature Anomaly Heatmap: Decades vs Year in Decade', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Year in Decade (0-9)')
        plt.ylabel('Decade')
        
        plt.tight_layout()
        plt.savefig('temperature_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Heatmap visualization created and saved!")
    
    def statistical_analysis(self):
        """
        Perform statistical analysis on the temperature data
        
        Teaching moment: Always back up visualizations with solid statistics
        """
        if self.processed_data is None:
            print("❌ Please process data first using process_data()")
            return
        
        print("📊 Performing statistical analysis...")
        
        df = self.processed_data
        
        # Linear trend analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['Year'], df['Temperature_Anomaly'])
        
        # Calculate warming rate per decade
        warming_per_decade = slope * 10
        
        # Recent trend (last 50 years)
        recent_df = df[df['Year'] >= df['Year'].max() - 50]
        recent_slope, _, recent_r, recent_p, _ = stats.linregress(recent_df['Year'], recent_df['Temperature_Anomaly'])
        recent_warming_per_decade = recent_slope * 10
        
        print(f"\n🔍 Statistical Analysis Results:")
        print(f"{'='*50}")
        print(f"Overall Trend ({df['Year'].min()}-{df['Year'].max()}):")
        print(f"  • Linear warming rate: {warming_per_decade:.4f}°C per decade")
        print(f"  • Correlation coefficient (R): {r_value:.4f}")
        print(f"  • Statistical significance (p-value): {p_value:.2e}")
        
        print(f"\nRecent Trend (Last 50 years):")
        print(f"  • Linear warming rate: {recent_warming_per_decade:.4f}°C per decade")
        print(f"  • Correlation coefficient (R): {recent_r:.4f}")
        print(f"  • Statistical significance (p-value): {recent_p:.2e}")
        
        # Temperature extremes analysis
        print(f"\nTemperature Extremes:")
        print(f"  • Warmest year: {df.loc[df['Temperature_Anomaly'].idxmax(), 'Year']} ({df['Temperature_Anomaly'].max():.3f}°C)")
        print(f"  • Coolest year: {df.loc[df['Temperature_Anomaly'].idxmin(), 'Year']} ({df['Temperature_Anomaly'].min():.3f}°C)")
        
        # Decade analysis
        decade_stats = df.groupby('Decade')['Temperature_Anomaly'].agg(['mean', 'std', 'count']).round(3)
        print(f"\nDecade-by-Decade Analysis:")
        print(decade_stats)
        
        return {
            'overall_trend': warming_per_decade,
            'recent_trend': recent_warming_per_decade,
            'correlation': r_value,
            'p_value': p_value
        }

def main():
    """
    Main function to run the complete analysis
    
    Teaching moment: Always structure your code with a clear main function
    """
    print("🌍 Earth's Surface Temperature Trend Visualization")
    print("=" * 55)
    print("This project demonstrates professional climate data analysis")
    print("and visualization techniques using Python.\n")
    
    # Initialize the visualizer
    visualizer = ClimateDataVisualizer()
    
    # Step 1: Try to download real NASA data
    print("Step 1: Data Acquisition")
    if not visualizer.download_nasa_data():
        print("Using sample data for demonstration...")
        visualizer.load_sample_data()
    
    # Step 2: Process the data
    print("\nStep 2: Data Processing")
    visualizer.process_data()
    
    # Step 3: Create visualizations
    print("\nStep 3: Creating Visualizations")
    visualizer.create_trend_visualization()
    visualizer.create_heatmap_visualization()
    
    # Step 4: Statistical analysis
    print("\nStep 4: Statistical Analysis")
    stats_results = visualizer.statistical_analysis()
    
    # Final summary
    print(f"\n🎉 Analysis Complete!")
    print(f"Generated files:")
    print(f"  • temperature_trends_comprehensive.png")
    print(f"  • temperature_heatmap.png")
    print(f"  • nasa_temp_data.csv (if download successful)")
    
    print(f"\n💡 Key Findings:")
    print(f"  • Overall warming trend: {stats_results['overall_trend']:.4f}°C per decade")
    print(f"  • Recent warming trend: {stats_results['recent_trend']:.4f}°C per decade")
    print(f"  • Statistical confidence: {stats_results['correlation']:.3f} correlation")

if __name__ == "__main__":
    main()
