import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests

# Set up plotting style
plt.style.use('default')
sns.set_palette("Set2")

print("🎓 Welcome to Climate Data Analysis Bootcamp!")
print("Follow these exercises to master climate data visualization\n")

# ============================================================================
# EXERCISE 1: BASIC DATA EXPLORATION
# ============================================================================

def exercise_1_data_exploration():
    """
    Exercise 1: Learn to load and explore climate data
    
    Learning Goals:
    - Load CSV data with pandas
    - Explore data structure and statistics
    - Handle missing values
    """
    print("📊 EXERCISE 1: Data Exploration Fundamentals")
    print("-" * 50)
    
    # Create sample temperature data for learning
    np.random.seed(42)
    years = range(1880, 2024)
    
    # Simulate realistic temperature data with trend + noise
    trend = np.linspace(-0.3, 1.2, len(years))
    noise = np.random.normal(0, 0.2, len(years))
    temp_anomaly = trend + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'Year': years,
        'Temperature_Anomaly': temp_anomaly,
        'Source': 'NASA_GISS'
    })
    
    # Add some missing values to make it realistic
    df.loc[5:10, 'Temperature_Anomaly'] = np.nan
    
    print("🔍 Step 1: Examine the data structure")
    print(f"Data shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}\n")
    
    print("🔍 Step 2: Look at the first few rows")
    print(df.head(10))
    print()
    
    print("🔍 Step 3: Basic statistics")
    print(df.describe())
    print()
    
    print("🔍 Step 4: Check for missing values")
    print(f"Missing values:\n{df.isnull().sum()}")
    print()
    
    print("✅ Exercise 1 Complete!")
    print("💡 Key Learning: Always explore your data before analysis!")
    print("   - Check data types and shapes")
    print("   - Look for missing values")
    print("   - Calculate basic statistics")
    print()
    
    return df

# ============================================================================
# EXERCISE 2: BASIC VISUALIZATION
# ============================================================================

def exercise_2_basic_plotting(df):
    """
    Exercise 2: Create your first climate visualization
    
    Learning Goals:
    - Create line plots with matplotlib
    - Add titles and labels
    - Save plots to files
    """
    print("📈 EXERCISE 2: Basic Climate Plotting")
    print("-" * 50)
    
    # Clean the data first (handle missing values)
    df_clean = df.dropna()
    
    print("🎨 Step 1: Create a simple line plot")
    
    # Create your first climate plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_clean['Year'], df_clean['Temperature_Anomaly'], 
             color='blue', linewidth=1.5, alpha=0.8)
    
    # Add horizontal reference line at zero
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Customize the plot
    plt.title('Global Temperature Anomalies (1880-2023)', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('exercise_2_basic_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Basic plot created and saved!")
    print()
    
    print("🎨 Step 2: Add color coding for warming/cooling")
    
    # Create a more sophisticated plot with color coding
    plt.figure(figsize=(12, 6))
    
    # Separate positive and negative anomalies
    positive_mask = df_clean['Temperature_Anomaly'] >= 0
    negative_mask = df_clean['Temperature_Anomaly'] < 0
    
    # Plot with different colors
    plt.plot(df_clean[positive_mask]['Year'], 
             df_clean[positive_mask]['Temperature_Anomaly'], 
             color='red', linewidth=1.5, alpha=0.8, label='Warming')
    plt.plot(df_clean[negative_mask]['Year'], 
             df_clean[negative_mask]['Temperature_Anomaly'], 
             color='blue', linewidth=1.5, alpha=0.8, label='Cooling')
    
    # Fill areas for better visual impact
    plt.fill_between(df_clean['Year'], 0, df_clean['Temperature_Anomaly'],
                     where=(df_clean['Temperature_Anomaly'] >= 0),
                     color='red', alpha=0.3)
    plt.fill_between(df_clean['Year'], 0, df_clean['Temperature_Anomaly'],
                     where=(df_clean['Temperature_Anomaly'] < 0),
                     color='blue', alpha=0.3)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    plt.title('Temperature Anomalies: Warming vs Cooling Periods', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('exercise_2_colored_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Exercise 2 Complete!")
    print("💡 Key Learning: Good visualizations tell a story!")
    print("   - Use colors meaningfully")
    print("   - Add reference lines for context")
    print("   - Always label axes and add titles")
    print()
    
    return df_clean

# ============================================================================
# EXERCISE 3: DATA PROCESSING AND TRENDS
# ============================================================================

def exercise_3_trend_analysis(df):
    """
    Exercise 3: Calculate and visualize trends
    
    Learning Goals:
    - Calculate moving averages
    - Perform linear regression
    - Create multi-line plots
    """
    print("📊 EXERCISE 3: Trend Analysis")
    print("-" * 50)
    
    print("🔄 Step 1: Calculate moving averages")
    
    # Calculate different moving averages
    df['MA_5yr'] = df['Temperature_Anomaly'].rolling(window=5, center=True).mean()
    df['MA_10yr'] = df['Temperature_Anomaly'].rolling(window=10, center=True).mean()
    df['MA_20yr'] = df['Temperature_Anomaly'].rolling(window=20, center=True).mean()
    
    print("Moving averages calculated:")
    print("- 5-year moving average")
    print("- 10-year moving average") 
    print("- 20-year moving average")
    print()
    
    print("📈 Step 2: Visualize trends")
    
    plt.figure(figsize=(14, 8))
    
    # Plot original data and moving averages
    plt.plot(df['Year'], df['Temperature_Anomaly'], 
             color='lightgray', alpha=0.7, linewidth=0.8, label='Annual Data')
    plt.plot(df['Year'], df['MA_5yr'], 
             color='blue', linewidth=1.5, label='5-year Average')
    plt.plot(df['Year'], df['MA_10yr'], 
             color='orange', linewidth=2, label='10-year Average')
    plt.plot(df['Year'], df['MA_20yr'], 
             color='red', linewidth=2.5, label='20-year Average')
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Temperature Trends: Raw Data vs Moving Averages', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('exercise_3_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("🔍 Step 3: Calculate linear trend")
    
    # Perform linear regression
    from scipy import stats
    
    # Remove NaN values for regression
    valid_data = df.dropna(subset=['Temperature_Anomaly'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        valid_data['Year'], valid_data['Temperature_Anomaly'])
    
    # Calculate trend line
    trend_line = slope * valid_data['Year'] + intercept
    
    # Convert slope to warming per decade
    warming_per_decade = slope * 10
    
    print(f"📊 Trend Analysis Results:")
    print(f"   • Warming rate: {warming_per_decade:.4f}°C per decade")
    print(f"   • Correlation (R): {r_value:.4f}")
    print(f"   • Statistical significance (p-value): {p_value:.2e}")
    print()
    
    # Plot with trend line
    plt.figure(figsize=(12, 8))
    plt.scatter(valid_data['Year'], valid_data['Temperature_Anomaly'], 
                alpha=0.6, color='blue', s=20, label='Annual Data')
    plt.plot(valid_data['Year'], trend_line, 
             color='red', linewidth=3, label=f'Linear Trend ({warming_per_decade:.3f}°C/decade)')
    plt.plot(valid_data['Year'], valid_data['MA_10yr'], 
             color='orange', linewidth=2, label='10-year Average')
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Temperature Trend Analysis with Linear Regression', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('exercise_3_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Exercise 3 Complete!")
    print("💡 Key Learning: Trends reveal the big picture!")
    print("   - Moving averages smooth out noise")
    print("   - Linear regression quantifies trends")
    print("   - Statistical tests validate findings")
    print()
    
    return df

# ============================================================================
# EXERCISE 4: ADVANCED VISUALIZATIONS
# ============================================================================

def exercise_4_advanced_viz(df):
    """
    Exercise 4: Create publication-quality visualizations
    
    Learning Goals:
    - Create subplot layouts
    - Make heatmaps
    - Add statistical annotations
    """
    print("🎨 EXERCISE 4: Advanced Visualizations")
    print("-" * 50)
    
    print("📊 Step 1: Create a comprehensive dashboard")
    
    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Climate Data Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Time series (top left)
    ax1 = axes[0, 0]
    ax1.plot(df['Year'], df['Temperature_Anomaly'], color='blue', alpha=0.7, linewidth=1)
    ax1.plot(df['Year'], df['MA_10yr'], color='red', linewidth=2, label='10-yr Average')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('Temperature Anomaly Time Series')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Temperature Anomaly (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram (top right)
    ax2 = axes[0, 1]
    ax2.hist(df['Temperature_Anomaly'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Anomaly')
    ax2.axvline(x=df['Temperature_Anomaly'].mean(), color='orange', linestyle='-', linewidth=2, 
                label=f'Mean ({df["Temperature_Anomaly"].mean():.2f}°C)')
    ax2.set_title('Distribution of Temperature Anomalies')
    ax2.set_xlabel('Temperature Anomaly (°C)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # Plot 3: Decade comparison (bottom left)
    ax3 = axes[1, 0]
    df['Decade'] = (df['Year'] // 10) * 10
    decade_avg = df.groupby('Decade')['Temperature_Anomaly'].mean()
    
    bars = ax3.bar(decade_avg.index, decade_avg.values, 
                   color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(decade_avg))))
    ax3.set_title('Average Temperature by Decade')
    ax3.set_xlabel('Decade')
    ax3.set_ylabel('Average Temperature Anomaly (°C)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, decade_avg.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Recent focus (bottom right)
    ax4 = axes[1, 1]
    recent_data = df[df['Year'] >= 1980]
    
    ax4.fill_between(recent_data['Year'], 0, recent_data['Temperature_Anomaly'],
                     where=(recent_data['Temperature_Anomaly'] >= 0),
                     color='red', alpha=0.4, label='Above Average')
    ax4.fill_between(recent_data['Year'], 0, recent_data['Temperature_Anomaly'],
                     where=(recent_data['Temperature_Anomaly'] < 0),
                     color='blue', alpha=0.4, label='Below Average')
    ax4.plot(recent_data['Year'], recent_data['MA_10yr'], 
             color='black', linewidth=2, label='10-yr Average')
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax4.set_title('Recent Trends (1980-Present)')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Temperature Anomaly (°C)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exercise_4_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Step 2: Create a temperature heatmap")
    
    # Prepare data for heatmap
    df['Year_Group'] = ((df['Year'] - 1880) // 10) * 10 + 1880
    df['Year_in_Group'] = df['Year'] - df['Year_Group']
    
    # Create pivot table for heatmap
    heatmap_data = df.pivot_table(values='Temperature_Anomaly',
                                  index='Year_Group',
                                  columns='Year_in_Group',
                                  aggfunc='mean')
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.2f', 
                cmap='RdYlBu_r',
                center=0,
                cbar_kws={'label': 'Temperature Anomaly (°C)'})
    
    plt.title('Temperature Anomaly Heatmap: Decade Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Year in Decade (0-9)')
    plt.ylabel('Decade Starting Year')
    
    plt.tight_layout()
    plt.savefig('exercise_4_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Exercise 4 Complete!")
    print("💡 Key Learning: Advanced visualizations reveal hidden patterns!")
    print("   - Subplots allow multiple perspectives")
    print("   - Heatmaps show two-dimensional patterns")
    print("   - Color coding enhances understanding")
    print()

# ============================================================================
# EXERCISE 5: INTERACTIVE ANALYSIS
# ============================================================================

def exercise_5_interactive_analysis():
    """
    Exercise 5: Practice with real NASA data
    
    Learning Goals:
    - Download real data from APIs
    - Handle API errors gracefully
    - Compare your analysis with real climate data
    """
    print("🌍 EXERCISE 5: Real NASA Data Analysis")
    print("-" * 50)
    
    print("📡 Step 1: Download real NASA GISS data")
    
    try:
        # Download real NASA data
        url = "https://data.giss.nasa.gov/gistemp/graphs/graph_data/Global_Mean_Estimates_based_on_Land_and_Ocean_Data/graph.txt"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse the data
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
        
        real_data = pd.DataFrame({
            'Year': years,
            'Temperature_Anomaly': temps
        })
        
        print(f"✅ Successfully downloaded {len(real_data)} years of NASA data!")
        print(f"Data range: {real_data['Year'].min()} - {real_data['Year'].max()}")
        
        # Quick analysis of real data
        recent_trend = real_data[real_data['Year'] >= 1980]['Temperature_Anomaly'].mean()
        print(f"Recent average (1980+): {recent_trend:.3f}°C")
        
        # Create a quick visualization
        plt.figure(figsize=(12, 6))
        plt.plot(real_data['Year'], real_data['Temperature_Anomaly'], 
                 color='blue', linewidth=1.5, alpha=0.8)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Real NASA GISS Global Temperature Anomalies', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Temperature Anomaly (°C)')
        plt.grid(True, alpha=0.3)
        
        # Highlight recent warming
        recent_data = real_data[real_data['Year'] >= 2000]
        plt.plot(recent_data['Year'], recent_data['Temperature_Anomaly'], 
                 color='red', linewidth=2, label='21st Century')
        plt.legend()
        
        plt.savefig('exercise_5_real_nasa_data.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("🎉 Congratulations! You've analyzed real climate data!")
        
    except Exception as e:
        print(f"❌ Could not download real data: {e}")
        print("💡 This is normal - APIs can be unreliable!")
        print("In real projects, always have backup plans.")
    
    print("\n✅ Exercise 5 Complete!")
    print("💡 Key Learning: Real-world data analysis involves:")
    print("   - Handling network errors gracefully")
    print("   - Parsing various data formats")
    print("   - Validating data quality")
    print()

# ============================================================================
# MAIN LEARNING PROGRAM
# ============================================================================

def run_learning_program():
    """
    Run the complete learning program
    """
    print("🎓 CLIMATE DATA ANALYSIS BOOTCAMP")
    print("=" * 55)
    print("Learn professional climate data analysis step by step!")
    print()
    
    # Exercise 1: Data Exploration
    df = exercise_1_data_exploration()
    
    input("Press Enter to continue to Exercise 2...")
    
    # Exercise 2: Basic Plotting
    df_clean = exercise_2_basic_plotting(df)
    
    input("Press Enter to continue to Exercise 3...")
    
    # Exercise 3: Trend Analysis
    df_processed = exercise_3_trend_analysis(df_clean)
    
    input("Press Enter to continue to Exercise 4...")
    
    # Exercise 4: Advanced Visualizations
    exercise_4_advanced_viz(df_processed)
    
    input("Press Enter to continue to Exercise 5...")
    
    # Exercise 5: Real Data Analysis
    exercise_5_interactive_analysis()
    
    print("\n🎉 CONGRATULATIONS!")
    print("=" * 55)
    print("You've completed the Climate Data Analysis Bootcamp!")
    print()
    print("🏆 Skills You've Mastered:")
    print("   ✅ Data loading and exploration")
    print("   ✅ Basic and advanced visualization")
    print("   ✅ Statistical trend analysis")
    print("   ✅ Professional plot creation")
    print("   ✅ Real-world data handling")
    print()
    print("🚀 Next Steps:")
    print("   • Try the full climate_visualizer.py script")
    print("   • Experiment with different datasets")
    print("   • Create your own analysis projects")
    print("   • Share your visualizations!")

if __name__ == "__main__":
    run_learning_program()
