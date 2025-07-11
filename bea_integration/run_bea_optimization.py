"""
Complete BEA Optimization Runner - All functions included
No external dependencies except your existing modules
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import folium

# Import your existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import spectrum_optimizer
from core import Spectrum_Optimizer_Result_Analyzer as analyzer
import bea_mapper  # FIXED: Changed from "from bea_integration import bea_mapper"

# Import the new BEA integration modules
from bea_spectrum_visualizer import BEASpectrumOptimizer
from bea_csv_reader import (
    read_and_analyze_bea_csv,
    visualize_bea_data,
    extract_geographic_hints,
    prepare_for_bea_mapping
)


def prepare_aws_stations(station_data, bea_geojson_path):
    """
    Prepare AWS (Advanced Wireless Service) stations for optimization
    """
    # Load BEA shapes to get geographic centers
    bea_shapes = gpd.read_file(bea_geojson_path)
    
    # Create optimization dataframe
    opt_df = pd.DataFrame()
    
    # Use callsigns as station IDs
    opt_df['station_id'] = station_data['callsigns']
    
    # AWS spectrum allocation strategy:
    # - Major carriers (based on licensee name patterns)
    # - Different bandwidth allocations based on licensee
    
    bandwidth_map = {
        'Liberty Mobile': 20,  # Major carrier
        'Gamma Acquisition': 15,  # Regional carrier
        'Default': 10  # Others
    }
    
    # Assign bandwidth based on licensee
    opt_df['bandwidth_mhz'] = 10  # Default
    for pattern, bw in bandwidth_map.items():
        if pattern != 'Default':
            mask = station_data['name'].str.contains(pattern, na=False)
            opt_df.loc[mask, 'bandwidth_mhz'] = bw
    
    # Generate coordinates based on callsign patterns
    # AWS licenses often have geographic indicators in callsigns
    opt_df['x_coord'] = np.nan
    opt_df['y_coord'] = np.nan
    
    # For demonstration, distribute stations across BEAs
    # In reality, you would use actual tower locations
    np.random.seed(42)
    
    # Get BEA centroids
    bea_shapes['centroid'] = bea_shapes.geometry.centroid
    bea_coords = [(geom.x, geom.y) for geom in bea_shapes['centroid']]
    
    # Distribute stations across BEAs with some clustering
    for i in range(len(opt_df)):
        # Select a BEA (weighted towards populated areas)
        bea_idx = np.random.choice(len(bea_coords))
        base_x, base_y = bea_coords[bea_idx]
        
        # Add some random offset (within ~50km)
        offset_x = np.random.normal(0, 0.5)  # ~50km at mid-latitudes
        offset_y = np.random.normal(0, 0.5)
        
        opt_df.loc[i, 'x_coord'] = base_x + offset_x
        opt_df.loc[i, 'y_coord'] = base_y + offset_y
    
    # Antenna parameters for AWS
    # AWS typically uses directional antennas
    opt_df['azimuth_deg'] = np.random.choice([0, 120, 240], len(opt_df))  # 3-sector sites
    opt_df['elevation_deg'] = np.random.uniform(-5, 5, len(opt_df))  # Slight downtilt
    
    # Assign to BEA regions
    station_gdf = gpd.GeoDataFrame(
        opt_df,
        geometry=gpd.points_from_xy(opt_df.x_coord, opt_df.y_coord),
        crs='EPSG:4326'
    )
    
    # Ensure BEA shapes have same CRS
    if bea_shapes.crs != station_gdf.crs:
        bea_shapes = bea_shapes.to_crs(station_gdf.crs)
    
    # Spatial join with BEA shapes
    stations_with_bea = gpd.sjoin(station_gdf, bea_shapes, how='left', predicate='within')
    
    opt_df['bea_id'] = stations_with_bea.get('bea', 'Unknown')
    opt_df['bea_name'] = stations_with_bea.get('Name', 'Unknown')
    
    # Handle stations outside BEA regions
    unassigned = opt_df['bea_id'] == 'Unknown'
    if unassigned.any():
        print(f"Warning: {unassigned.sum()} stations could not be assigned to BEA regions")
        # Assign to nearest BEA
        opt_df.loc[unassigned, 'bea_id'] = 'BEA_999'
        opt_df.loc[unassigned, 'bea_name'] = 'Unassigned'
    
    # Create clusters within BEAs
    # Group nearby stations into clusters for optimization
    opt_df['cluster'] = opt_df['bea_id'].astype(str) + '_0'  # Simple clustering
    
    # Extract state from BEA name (e.g., "Bangor, ME" -> "ME")
    def extract_state_from_name(name):
        if pd.isna(name) or name == 'Unknown':
            return 'XX'
        # Most BEA names end with ", STATE"
        parts = name.split(', ')
        if len(parts) >= 2:
            # Get the last part which should be the state
            state = parts[-1].strip()
            # Check if it's a 2-letter state code
            if len(state) == 2 and state.isalpha():
                return state.upper()
        return 'XX'  # Unknown state
    
    opt_df['state'] = opt_df['bea_name'].apply(extract_state_from_name)
    
    # Area type based on station density
    bea_station_counts = opt_df['bea_id'].value_counts()
    urban_beas = bea_station_counts[bea_station_counts > 10].index
    rural_beas = bea_station_counts[bea_station_counts < 3].index
    
    opt_df['area_type'] = 'suburban'  # Default
    opt_df.loc[opt_df['bea_id'].isin(urban_beas), 'area_type'] = 'urban'
    opt_df.loc[opt_df['bea_id'].isin(rural_beas), 'area_type'] = 'rural'
    
    # Add licensee information
    opt_df['licensee'] = station_data['name']
    opt_df['frn'] = station_data['FRN']
    opt_df['expiration_date'] = station_data['expiration date']
    
    return opt_df


def create_aws_visualizations(results, output_dir):
    """
    Create AWS-specific visualizations
    """
    # 1. Licensee spectrum allocation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Group by licensee
    licensee_stats = results.groupby('licensee').agg({
        'station_id': 'count',
        'bandwidth_mhz': 'sum',
        'optimized_start_freq_mhz': lambda x: x.max() - x.min() if x.notna().any() else 0
    }).rename(columns={
        'station_id': 'station_count',
        'bandwidth_mhz': 'total_bandwidth',
        'optimized_start_freq_mhz': 'spectrum_span'
    })
    
    # Top 10 licensees by station count
    top_licensees = licensee_stats.nlargest(10, 'station_count')
    
    # Station count
    top_licensees['station_count'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_ylabel('Number of Stations')
    ax1.set_title('Top 10 AWS Licensees by Station Count', fontsize=14, weight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Total bandwidth
    top_licensees['total_bandwidth'].plot(kind='bar', ax=ax2, color='lightcoral')
    ax2.set_ylabel('Total Bandwidth (MHz)')
    ax2.set_title('Top 10 AWS Licensees by Total Bandwidth', fontsize=14, weight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aws_licensee_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Spectrum efficiency heatmap by BEA
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate efficiency by BEA
    bea_efficiency = results.groupby('bea_name').apply(
        lambda x: (x['bandwidth_mhz'].sum() /
                  (x['optimized_end_freq_mhz'].max() - x['optimized_start_freq_mhz'].min()) * 100)
        if x['optimized_start_freq_mhz'].notna().any() else 0
    ).sort_values(ascending=False).head(20)
    
    # Create heatmap data
    bea_efficiency.plot(kind='barh', ax=ax, color='green', alpha=0.7)
    ax.set_xlabel('Spectrum Efficiency (%)')
    ax.set_title('AWS Spectrum Efficiency by BEA Region (Top 20)', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aws_bea_efficiency.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Optimization method success rate
    fig, ax = plt.subplots(figsize=(10, 10))
    
    method_counts = results['allocation_method'].value_counts()
    colors = plt.cm.Set3(range(len(method_counts)))
    
    wedges, texts, autotexts = ax.pie(method_counts.values,
                                      labels=method_counts.index,
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90)
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_weight('bold')
        autotext.set_fontsize(10)
    
    ax.set_title('AWS Spectrum Allocation Methods', fontsize=16, weight='bold')
    
    # Add legend with counts
    legend_labels = [f'{method}: {count} stations' for method, count in method_counts.items()]
    ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aws_allocation_methods.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created AWS-specific visualizations in {output_dir}/")


def main():
    """
    Main function to run the complete BEA optimization workflow
    """
    print("=" * 80)
    print("BEA SPECTRUM OPTIMIZATION SYSTEM")
    print("=" * 80)
    print()
    
    # Configuration - Update these paths as needed
    config = {
        'bea_geojson': '../data/bea.geojson',
        'bea_csv': '../data/example_bea_table.csv',
        'output_dir': 'bea_optimization_output',
        'use_partitioning': True,
        'plot_results': True
    }
    
    # Create output directory
    Path(config['output_dir']).mkdir(exist_ok=True)
    
    try:
        # Step 1: Analyze the BEA CSV file
        print("\n📊 STEP 1: Analyzing BEA CSV file...")
        print("-" * 60)
        bea_data = read_and_analyze_bea_csv(config['bea_csv'])
        
        # Create initial visualizations of the raw data
        visualize_bea_data(bea_data, config['output_dir'])
        
        # Extract geographic hints
        geo_hints = extract_geographic_hints(bea_data)
        
        # Step 2: Initialize BEA Spectrum Optimizer
        print("\n🔧 STEP 2: Initializing BEA Spectrum Optimizer...")
        print("-" * 60)
        optimizer = BEASpectrumOptimizer(config['bea_geojson'], config['bea_csv'])
        optimizer.load_bea_data()
        
        # Step 3: Prepare optimization input
        print("\n📍 STEP 3: Preparing station data for optimization...")
        print("-" * 60)
        
        # For AWS stations, use specialized preparation
        optimization_df = prepare_aws_stations(bea_data, config['bea_geojson'])
        
        # Save prepared data
        input_path = f"{config['output_dir']}/optimization_input.csv"
        optimization_df.to_csv(input_path, index=False)
        print(f"✅ Saved {len(optimization_df)} stations to {input_path}")
        
        # Step 4: Run spectrum optimization
        print("\n🚀 STEP 4: Running spectrum optimization...")
        print("-" * 60)
        
        output_path = f"{config['output_dir']}/optimized_spectrum.csv"
        
        # Option A: Use the integrated optimizer (recommended)
        results = optimizer.run_optimization(
            input_csv=input_path,
            output_csv=output_path
        )
        
        # Option B: Use your original optimizer directly
        # results = spectrum_optimizer.run_optimizer(
        #     input_csv=input_path,
        #     output_csv=output_path,
        #     use_partitioning=config['use_partitioning'],
        #     plot_results=False  # We'll do custom plotting
        # )
        
        if results is None:
            print("\n❌ Optimization failed! Check the logs above.")
            return 1
        
        # Step 5: Analyze optimization results
        print("\n📈 STEP 5: Analyzing optimization results...")
        print("-" * 60)
        
        # Use your existing analyzer
        analyzer.analyze_optimization_results(output_path)
        
        # Create result visualizations
        analyzer.visualize_spectrum_allocation(
            results,
            save_path=f"{config['output_dir']}/spectrum_visualization.png"
        )
        
        # Find potential issues
        issues = analyzer.find_optimization_issues(results)
        
        # Step 6: Create geographic visualizations
        print("\n🗺️  STEP 6: Creating geographic visualizations...")
        print("-" * 60)
        
        # Create interactive map
        map_path = f"{config['output_dir']}/interactive_spectrum_map.html"
        optimizer.create_geographic_visualization(map_path)
        print(f"✅ Created interactive map: {map_path}")
        
        # Create static visualizations
        optimizer.create_static_visualizations(config['output_dir'])
        
        # Create AWS-specific visualizations
        create_aws_visualizations(results, config['output_dir'])
        
        # Step 7: Generate comprehensive report
        print("\n📄 STEP 7: Generating comprehensive report...")
        print("-" * 60)
        
        report_dir = f"{config['output_dir']}/full_report"
        optimizer.generate_full_report(report_dir)
        
        # Step 8: Create BEA mapping outputs
        print("\n🌍 STEP 8: Creating BEA mapping outputs...")
        print("-" * 60)
        
        # Merge optimization results with BEA shapes for further analysis
        bea_shapes = bea_mapper.load_bea_shapes(config['bea_geojson'])
        
        # Create BEA summary data
        results_by_bea = results.groupby('bea_id').agg({
            'station_id': 'count',
            'bandwidth_mhz': 'sum',
            'optimized_start_freq_mhz': 'min',
            'optimized_end_freq_mhz': 'max'
        }).reset_index()
        results_by_bea.columns = ['bea', 'station_count', 'total_bandwidth',
                                  'min_freq', 'max_freq']
        
        # Save BEA summary first
        bea_summary_path = f"{config['output_dir']}/bea_summary.csv"
        results_by_bea.to_csv(bea_summary_path, index=False)
        
        # Merge with BEA shapes
        merged_gdf = bea_mapper.merge_bea_data(
            bea_shapes,
            bea_summary_path,
            join_field="bea"
        )
        
        # Save merged outputs
        bea_mapper.save_merged_geojson(
            merged_gdf,
            f"{config['output_dir']}/bea_with_spectrum_data.geojson"
        )
        bea_mapper.save_merged_csv(
            merged_gdf,
            f"{config['output_dir']}/bea_with_spectrum_data.csv"
        )
        
        # Final summary
        print("\n" + "=" * 80)
        print("✅ BEA SPECTRUM OPTIMIZATION COMPLETE!")
        print("=" * 80)
        print(f"\n📁 All results saved to: {config['output_dir']}/")
        print("\nKey outputs:")
        print(f"  • Interactive map: {map_path}")
        print(f"  • Optimization results: {output_path}")
        print(f"  • Full report: {report_dir}/")
        print(f"  • BEA GeoJSON with data: {config['output_dir']}/bea_with_spectrum_data.geojson")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Run the main function
    exit_code = main()
    sys.exit(exit_code)
