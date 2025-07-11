import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration constants needed for analysis
CHANNEL_STEP = 5  # Default channel step in MHz

def analyze_optimization_results(csv_path):
    """
    Comprehensive analysis of spectrum optimization results.
    """
    # Load results
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("SPECTRUM OPTIMIZATION RESULTS ANALYSIS")
    print("=" * 80)
    
    # Basic statistics
    print("\n📊 BASIC STATISTICS:")
    print(f"Total stations optimized: {len(df)}")
    print(f"Total bandwidth allocated: {df['bandwidth_mhz'].sum()} MHz")
    
    # Allocation method breakdown
    print("\n🔧 ALLOCATION METHOD BREAKDOWN:")
    method_counts = df['allocation_method'].value_counts()
    for method, count in method_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {method}: {count} stations ({percentage:.1f}%)")
    
    # Spectrum efficiency
    freq_min = df['optimized_start_freq_mhz'].min()
    freq_max = df['optimized_end_freq_mhz'].max()
    total_span = freq_max - freq_min
    total_bandwidth = df['bandwidth_mhz'].sum()
    efficiency = (total_bandwidth / total_span) * 100
    
    print(f"\n📡 SPECTRUM EFFICIENCY:")
    print(f"  Frequency range: {freq_min} - {freq_max} MHz")
    print(f"  Total spectrum span: {total_span} MHz")
    print(f"  Total bandwidth demand: {total_bandwidth} MHz")
    print(f"  Spectrum efficiency: {efficiency:.1f}%")
    
    # State-level analysis
    print("\n🗺️  STATE-LEVEL ANALYSIS:")
    state_stats = df.groupby('state').agg({
        'station_id': 'count',
        'bandwidth_mhz': 'sum',
        'optimized_start_freq_mhz': 'min',
        'optimized_end_freq_mhz': 'max'
    }).rename(columns={'station_id': 'station_count'})
    
    state_stats['spectrum_span'] = state_stats['optimized_end_freq_mhz'] - state_stats['optimized_start_freq_mhz']
    state_stats['efficiency'] = (state_stats['bandwidth_mhz'] / state_stats['spectrum_span'] * 100).round(1)
    
    print(state_stats.sort_values('station_count', ascending=False).head(10))
    
    # Cluster analysis
    print("\n🏘️  CLUSTER ANALYSIS:")
    cluster_methods = df.groupby(['cluster', 'allocation_method']).size().unstack(fill_value=0)
    print(cluster_methods)
    
    # Frequency reuse analysis
    print("\n🔄 FREQUENCY REUSE ANALYSIS:")
    
    # Count overlapping frequencies
    overlaps = 0
    reuse_pairs = []
    
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            start_i = df.iloc[i]['optimized_start_freq_mhz']
            end_i = df.iloc[i]['optimized_end_freq_mhz']
            start_j = df.iloc[j]['optimized_start_freq_mhz']
            end_j = df.iloc[j]['optimized_end_freq_mhz']
            
            # Check if frequencies overlap
            if start_i < end_j and start_j < end_i:
                overlaps += 1
                reuse_pairs.append((
                    df.iloc[i]['station_id'],
                    df.iloc[j]['station_id'],
                    df.iloc[i]['cluster'],
                    df.iloc[j]['cluster']
                ))
    
    print(f"  Total frequency overlaps: {overlaps}")
    print(f"  Reuse rate: {(overlaps / (len(df) * (len(df) - 1) / 2) * 100):.2f}%")
    
    # Show some reuse examples
    if reuse_pairs:
        print("\n  Example frequency reuse pairs:")
        for i, (s1, s2, c1, c2) in enumerate(reuse_pairs[:5]):
            print(f"    {s1} ({c1}) ↔ {s2} ({c2})")
        if len(reuse_pairs) > 5:
            print(f"    ... and {len(reuse_pairs) - 5} more pairs")
    
    # Bandwidth distribution
    print("\n📊 BANDWIDTH DISTRIBUTION:")
    bandwidth_stats = df['bandwidth_mhz'].describe()
    print(bandwidth_stats)
    
    # Area type performance
    print("\n🏙️  AREA TYPE PERFORMANCE:")
    area_stats = df.groupby('area_type').agg({
        'station_id': 'count',
        'bandwidth_mhz': ['sum', 'mean'],
        'allocation_method': lambda x: x.value_counts().index[0]
    })
    print(area_stats)
    
    return df


def visualize_spectrum_allocation(df, save_path='spectrum_visualization.png'):
    """
    Create comprehensive visualization of spectrum allocation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Spectrum usage by state
    ax1 = axes[0, 0]
    state_spectrum = df.groupby('state').agg({
        'optimized_start_freq_mhz': 'min',
        'optimized_end_freq_mhz': 'max'
    })
    state_spectrum['span'] = state_spectrum['optimized_end_freq_mhz'] - state_spectrum['optimized_start_freq_mhz']
    state_spectrum.nlargest(10, 'span')['span'].plot(kind='bar', ax=ax1)
    ax1.set_title('Top 10 States by Spectrum Usage')
    ax1.set_ylabel('Spectrum Span (MHz)')
    ax1.set_xlabel('State')
    
    # 2. Allocation method distribution
    ax2 = axes[0, 1]
    df['allocation_method'].value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%')
    ax2.set_title('Allocation Method Distribution')
    ax2.set_ylabel('')
    
    # 3. Bandwidth distribution
    ax3 = axes[1, 0]
    df['bandwidth_mhz'].hist(bins=20, ax=ax3, edgecolor='black')
    ax3.set_title('Bandwidth Requirements Distribution')
    ax3.set_xlabel('Bandwidth (MHz)')
    ax3.set_ylabel('Number of Stations')
    
    # 4. Frequency allocation timeline
    ax4 = axes[1, 1]
    # Sort by start frequency and plot as horizontal bars
    df_sorted = df.sort_values('optimized_start_freq_mhz').head(30)  # Show first 30
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax4.barh(i, 
                row['optimized_end_freq_mhz'] - row['optimized_start_freq_mhz'],
                left=row['optimized_start_freq_mhz'],
                height=0.8,
                alpha=0.7,
                color='blue' if row['allocation_method'] == 'Local_CP' else 'green')
    
    ax4.set_title('Frequency Allocation Sample (First 30 Stations)')
    ax4.set_xlabel('Frequency (MHz)')
    ax4.set_ylabel('Station Index')
    ax4.set_ylim(-1, 30)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Visualization saved to: {save_path}")


def find_optimization_issues(df, channel_step=5):
    """
    Identify potential issues or inefficiencies in the optimization.
    """
    print("\n⚠️  POTENTIAL OPTIMIZATION ISSUES:")
    
    issues = []
    
    # Check for very low efficiency clusters
    cluster_stats = df.groupby('cluster').agg({
        'bandwidth_mhz': 'sum',
        'optimized_start_freq_mhz': 'min',
        'optimized_end_freq_mhz': 'max'
    })
    cluster_stats['efficiency'] = (cluster_stats['bandwidth_mhz'] / 
                                  (cluster_stats['optimized_end_freq_mhz'] - 
                                   cluster_stats['optimized_start_freq_mhz'])) * 100
    
    low_efficiency = cluster_stats[cluster_stats['efficiency'] < 50]
    if not low_efficiency.empty:
        issues.append(f"Found {len(low_efficiency)} clusters with <50% efficiency")
        print(f"\n  Low efficiency clusters:")
        for cluster in low_efficiency.index:
            print(f"    {cluster}: {low_efficiency.loc[cluster, 'efficiency']:.1f}% efficiency")
    
    # Check for large bandwidth stations
    large_bw = df[df['bandwidth_mhz'] > 20]
    if not large_bw.empty:
        issues.append(f"Found {len(large_bw)} stations with >20MHz bandwidth")
        print(f"\n  Large bandwidth stations: {list(large_bw['station_id'].head())}")
    
    # Check for frequency gaps
    df_sorted = df.sort_values('optimized_start_freq_mhz')
    gaps = []
    for i in range(len(df_sorted) - 1):
        gap_start = df_sorted.iloc[i]['optimized_end_freq_mhz']
        gap_end = df_sorted.iloc[i + 1]['optimized_start_freq_mhz']
        if gap_end > gap_start + channel_step:  # Significant gap
            gaps.append((gap_start, gap_end, gap_end - gap_start))
    
    if gaps:
        issues.append(f"Found {len(gaps)} frequency gaps")
        print(f"\n  Frequency gaps found:")
        for start, end, size in sorted(gaps, key=lambda x: x[2], reverse=True)[:5]:
            print(f"    {start}-{end} MHz ({size} MHz gap)")
    
    if not issues:
        print("  No major issues detected!")
    else:
        print(f"\n  Summary: {len(issues)} types of issues found")
    
    return issues


# Example usage
if __name__ == "__main__":
    # Analyze the results
    results_df = analyze_optimization_results('optimized_spectrum_improved.csv')
    
    # Create visualizations
    visualize_spectrum_allocation(results_df)
    
    # Find potential issues (passing channel_step parameter)
    find_optimization_issues(results_df, channel_step=CHANNEL_STEP)
