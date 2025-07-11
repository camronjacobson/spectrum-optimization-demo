"""
BEA Spectrum Optimizer with Geographic Visualization
Integrates spectrum optimization with BEA region mapping and visualization
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
import folium
from folium import plugins
import json
import logging
from pathlib import Path

# Import your existing modules
from core import spectrum_optimizer
import bea_mapper
from core.Spectrum_Optimizer_Result_Analyzer import analyze_optimization_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BEASpectrumOptimizer:
    """
    Integrates BEA geographic data with spectrum optimization
    """
    
    def __init__(self, bea_geojson_path, bea_table_csv_path):
        """
        Initialize with BEA shape files and station data
        
        Parameters:
        -----------
        bea_geojson_path : str
            Path to BEA GeoJSON file with shape boundaries
        bea_table_csv_path : str
            Path to CSV with station/license data
        """
        self.bea_geojson_path = bea_geojson_path
        self.bea_table_csv_path = bea_table_csv_path
        self.bea_shapes = None
        self.station_data = None
        self.optimization_results = None
        
    def load_bea_data(self):
        """Load BEA shapes and station data"""
        logger.info("Loading BEA geographic data...")
        
        # Load BEA shapes
        self.bea_shapes = gpd.read_file(self.bea_geojson_path)
        logger.info(f"Loaded {len(self.bea_shapes)} BEA regions")
        
        # Load station data from CSV
        self.station_data = pd.read_csv(self.bea_table_csv_path)
        logger.info(f"Loaded {len(self.station_data)} stations from BEA table")
        
        # Display column information
        logger.info("Station data columns:")
        for col in self.station_data.columns:
            logger.info(f"  - {col}: {self.station_data[col].dtype}")
            
        return self.station_data
    
    def prepare_optimization_input(self, output_path="station_optimization_input.csv"):
        """
        Prepare station data for spectrum optimization
        Converts BEA table format to optimizer input format
        """
        logger.info("Preparing data for spectrum optimization...")
        
        # Create synthetic coordinates for demonstration
        # In practice, you would have actual lat/lon from FCC data
        optimization_df = pd.DataFrame()
        
        # Use callsigns as station IDs
        optimization_df['station_id'] = self.station_data['callsigns']
        
        # Generate synthetic coordinates within US bounds
        # You would replace this with actual coordinates
        np.random.seed(42)  # For reproducibility
        optimization_df['x_coord'] = np.random.uniform(-125, -65, len(self.station_data))  # Longitude
        optimization_df['y_coord'] = np.random.uniform(25, 50, len(self.station_data))     # Latitude
        
        # Assign bandwidth based on radio service type
        service_bandwidth_map = {
            'AM': 10,   # AM radio
            'FM': 0.2,  # FM radio
            'TV': 6,    # TV broadcast
            'LM': 25,   # Land Mobile
            'MW': 20,   # Microwave
        }
        
        # Extract service type from 'radio service' column
        optimization_df['bandwidth_mhz'] = 10  # Default
        if 'radio service' in self.station_data.columns:
            for service, bw in service_bandwidth_map.items():
                mask = self.station_data['radio service'].str.contains(service, na=False)
                optimization_df.loc[mask, 'bandwidth_mhz'] = bw
        
        # Add antenna parameters
        optimization_df['azimuth_deg'] = np.random.uniform(0, 360, len(self.station_data))
        optimization_df['elevation_deg'] = np.random.uniform(-10, 10, len(self.station_data))
        
        # Assign to BEA regions based on coordinates
        optimization_df = self._assign_bea_regions(optimization_df)
        
        # Add cluster information (BEA_ID + sub-cluster)
        optimization_df['cluster'] = optimization_df['bea_id'].astype(str) + '_0'
        
        # Add state from BEA name
        optimization_df['state'] = optimization_df['bea_name'].str[:2]
        
        # Add area type based on BEA characteristics
        optimization_df['area_type'] = 'suburban'  # Default
        # Major metro areas
        urban_beas = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
        rural_keywords = ['Rural', 'Non-Metro', 'Micropolitan']
        
        for urban in urban_beas:
            mask = optimization_df['bea_name'].str.contains(urban, na=False)
            optimization_df.loc[mask, 'area_type'] = 'urban'
            
        for rural in rural_keywords:
            mask = optimization_df['bea_name'].str.contains(rural, na=False)
            optimization_df.loc[mask, 'area_type'] = 'rural'
        
        # Save prepared data
        optimization_df.to_csv(output_path, index=False)
        logger.info(f"Saved optimization input to {output_path}")
        
        return optimization_df
    
    def _assign_bea_regions(self, df):
        """Assign stations to BEA regions based on coordinates"""
        logger.info("Assigning stations to BEA regions...")
        
        # Convert station dataframe to GeoDataFrame
        station_gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.x_coord, df.y_coord),
            crs='EPSG:4326'
        )
        
        # Ensure BEA shapes have same CRS
        if self.bea_shapes.crs != station_gdf.crs:
            self.bea_shapes = self.bea_shapes.to_crs(station_gdf.crs)
        
        # Spatial join to assign BEA regions
        stations_with_bea = gpd.sjoin(
            station_gdf,
            self.bea_shapes,
            how='left',
            predicate='within'
        )
        
        # Extract BEA information
        df['bea_id'] = stations_with_bea.get('bea', 'Unknown')
        df['bea_name'] = stations_with_bea.get('Name', 'Unknown')
        
        # Handle stations outside BEA regions
        unassigned = df['bea_id'] == 'Unknown'
        if unassigned.any():
            logger.warning(f"{unassigned.sum()} stations could not be assigned to BEA regions")
            # Assign to nearest BEA
            df.loc[unassigned, 'bea_id'] = 'BEA_999'
            df.loc[unassigned, 'bea_name'] = 'Unassigned'
        
        return df
    
    def run_optimization(self, input_csv=None, output_csv="bea_optimized_spectrum.csv"):
        """Run spectrum optimization on prepared data"""
        if input_csv is None:
            input_csv = "station_optimization_input.csv"
            
        logger.info("Running spectrum optimization...")
        
        # Use the existing spectrum optimizer
        self.optimization_results = spectrum_optimizer.run_optimizer(
            input_csv=input_csv,
            output_csv=output_csv,
            use_partitioning=True,
            plot_results=False  # We'll do custom plotting
        )
        
        return self.optimization_results
    
    def create_geographic_visualization(self, output_path="bea_spectrum_map.html"):
        """Create interactive map showing optimized spectrum allocation by BEA region"""
        if self.optimization_results is None:
            logger.error("No optimization results available. Run optimization first.")
            return
            
        logger.info("Creating geographic visualization...")
        
        # Create base map centered on US
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        
        # Merge optimization results with BEA data
        results_by_bea = self._aggregate_results_by_bea()
        
        # Create choropleth layer for spectrum efficiency
        if 'efficiency' in results_by_bea.columns:
            folium.Choropleth(
                geo_data=self.bea_shapes,
                name='Spectrum Efficiency',
                data=results_by_bea,
                columns=['bea_id', 'efficiency'],
                key_on='feature.properties.bea',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='Spectrum Efficiency (%)'
            ).add_to(m)
        
        # Add station markers with optimization info
        self._add_station_markers(m)
        
        # Add BEA boundaries
        folium.GeoJson(
            self.bea_shapes,
            name='BEA Boundaries',
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['bea', 'Name'],
                aliases=['BEA ID:', 'BEA Name:']
            )
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save(output_path)
        logger.info(f"Saved interactive map to {output_path}")
        
        return m
    
    def _aggregate_results_by_bea(self):
        """Aggregate optimization results by BEA region"""
        results_by_bea = self.optimization_results.groupby('bea_id').agg({
            'station_id': 'count',
            'bandwidth_mhz': 'sum',
            'optimized_start_freq_mhz': 'min',
            'optimized_end_freq_mhz': 'max'
        }).reset_index()
        
        results_by_bea.columns = ['bea_id', 'station_count', 'total_bandwidth',
                                  'min_freq', 'max_freq']
        
        # Calculate efficiency
        results_by_bea['spectrum_span'] = results_by_bea['max_freq'] - results_by_bea['min_freq']
        results_by_bea['efficiency'] = (results_by_bea['total_bandwidth'] /
                                       results_by_bea['spectrum_span'] * 100)
        results_by_bea['efficiency'] = results_by_bea['efficiency'].fillna(0)
        
        return results_by_bea
    
    def _add_station_markers(self, folium_map):
        """Add station markers to the map"""
        # Create marker clusters for better performance
        marker_cluster = plugins.MarkerCluster(name='Stations').add_to(folium_map)
        
        # Color stations by allocation method
        color_map = {
            'Local_CP': 'blue',
            'Global_Optimization': 'green',
            'Greedy_Heuristic': 'orange',
            'Extended_Spectrum': 'red',
            'No_Conflicts': 'gray',
            'Failed': 'darkred'
        }
        
        for idx, station in self.optimization_results.iterrows():
            if pd.isna(station['optimized_start_freq_mhz']):
                color = 'darkred'
                popup_text = f"""
                <b>Station:</b> {station['station_id']}<br>
                <b>BEA:</b> {station.get('bea_name', 'Unknown')}<br>
                <b>Status:</b> FAILED ALLOCATION<br>
                <b>Bandwidth Required:</b> {station['bandwidth_mhz']} MHz
                """
            else:
                color = color_map.get(station['allocation_method'], 'purple')
                popup_text = f"""
                <b>Station:</b> {station['station_id']}<br>
                <b>BEA:</b> {station.get('bea_name', 'Unknown')}<br>
                <b>Frequency:</b> {station['optimized_start_freq_mhz']:.1f} -
                                  {station['optimized_end_freq_mhz']:.1f} MHz<br>
                <b>Bandwidth:</b> {station['bandwidth_mhz']} MHz<br>
                <b>Method:</b> {station['allocation_method']}<br>
                <b>Area Type:</b> {station['area_type']}
                """
            
            folium.CircleMarker(
                location=[station['y_coord'], station['x_coord']],
                radius=5,
                popup=popup_text,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(marker_cluster)
    
    def create_static_visualizations(self, output_dir="bea_visualizations"):
        """Create static visualizations of the optimization results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. BEA Region Spectrum Usage Map
        self._plot_bea_spectrum_usage(output_dir)
        
        # 2. Station Distribution by BEA
        self._plot_station_distribution(output_dir)
        
        # 3. Spectrum Allocation Timeline by BEA
        self._plot_spectrum_timeline(output_dir)
        
        # 4. Optimization Method Success by Region
        self._plot_optimization_methods(output_dir)
        
    def _plot_bea_spectrum_usage(self, output_dir):
        """Plot spectrum usage efficiency by BEA region"""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        results_by_bea = self._aggregate_results_by_bea()
        
        # Merge with shapes
        bea_plot = self.bea_shapes.merge(results_by_bea,
                                         left_on='bea',
                                         right_on='bea_id',
                                         how='left')
        
        # Plot efficiency
        bea_plot.plot(column='efficiency',
                      cmap='RdYlGn',
                      linewidth=0.8,
                      ax=ax,
                      edgecolor='0.8',
                      legend=True,
                      legend_kwds={'label': 'Spectrum Efficiency (%)',
                                   'orientation': 'horizontal'})
        
        ax.set_title('Spectrum Usage Efficiency by BEA Region', fontsize=16, weight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/bea_spectrum_efficiency.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_station_distribution(self, output_dir):
        """Plot station count and bandwidth by BEA"""
        results_by_bea = self._aggregate_results_by_bea()
        top_beas = results_by_bea.nlargest(15, 'station_count')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Station count
        ax1.bar(range(len(top_beas)), top_beas['station_count'])
        ax1.set_xticks(range(len(top_beas)))
        ax1.set_xticklabels(top_beas['bea_id'], rotation=45, ha='right')
        ax1.set_ylabel('Number of Stations')
        ax1.set_title('Top 15 BEAs by Station Count', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Total bandwidth
        ax2.bar(range(len(top_beas)), top_beas['total_bandwidth'], color='orange')
        ax2.set_xticks(range(len(top_beas)))
        ax2.set_xticklabels(top_beas['bea_id'], rotation=45, ha='right')
        ax2.set_ylabel('Total Bandwidth (MHz)')
        ax2.set_title('Top 15 BEAs by Total Bandwidth Demand', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/bea_station_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_spectrum_timeline(self, output_dir):
        """Plot spectrum allocation timeline for top BEAs"""
        # Get top 5 BEAs by station count
        results_by_bea = self._aggregate_results_by_bea()
        top_5_beas = results_by_bea.nlargest(5, 'station_count')['bea_id'].tolist()
        
        fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
        
        for idx, (ax, bea_id) in enumerate(zip(axes, top_5_beas)):
            # Get stations in this BEA
            bea_stations = self.optimization_results[
                self.optimization_results['bea_id'] == bea_id
            ].copy()
            
            # Skip if no valid assignments
            valid_stations = bea_stations[bea_stations['optimized_start_freq_mhz'].notna()]
            if valid_stations.empty:
                ax.text(0.5, 0.5, f'No valid assignments for {bea_id}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel(bea_id)
                continue
                
            # Sort by frequency
            valid_stations = valid_stations.sort_values('optimized_start_freq_mhz')
            
            # Plot frequency blocks
            for i, (_, station) in enumerate(valid_stations.iterrows()):
                width = station['optimized_end_freq_mhz'] - station['optimized_start_freq_mhz']
                color = plt.cm.tab20(i % 20)
                
                ax.barh(0, width, left=station['optimized_start_freq_mhz'],
                       height=0.8, color=color, alpha=0.7, edgecolor='black')
                
                # Add station label if space permits
                if width > 5:
                    ax.text(station['optimized_start_freq_mhz'] + width/2, 0,
                           station['station_id'][:8], ha='center', va='center',
                           fontsize=8, rotation=90)
            
            ax.set_ylim(-0.5, 0.5)
            ax.set_ylabel(f"{bea_id}\n({len(valid_stations)} stations)")
            ax.set_yticks([])
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add BEA name as title
            bea_name = self.bea_shapes[self.bea_shapes['bea'] == bea_id]['Name'].iloc[0]
            ax.text(0.02, 0.95, bea_name, transform=ax.transAxes, fontsize=10,
                   va='top', style='italic')
        
        axes[-1].set_xlabel('Frequency (MHz)', fontsize=12)
        fig.suptitle('Spectrum Allocation Timeline - Top 5 BEAs', fontsize=16, weight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/bea_spectrum_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_optimization_methods(self, output_dir):
        """Plot optimization method distribution across BEAs"""
        # Get method counts by BEA
        method_by_bea = self.optimization_results.groupby(['bea_id', 'allocation_method']).size().unstack(fill_value=0)
        
        # Get top 10 BEAs
        top_beas = self.optimization_results['bea_id'].value_counts().head(10).index
        method_by_bea_top = method_by_bea.loc[top_beas]
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        method_by_bea_top.plot(kind='bar', stacked=True, ax=ax,
                               colormap='tab20', width=0.8)
        
        ax.set_xlabel('BEA Region', fontsize=12)
        ax.set_ylabel('Number of Stations', fontsize=12)
        ax.set_title('Optimization Methods by BEA Region (Top 10)', fontsize=14, weight='bold')
        ax.legend(title='Allocation Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/bea_optimization_methods.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_full_report(self, output_dir="bea_optimization_report"):
        """Generate comprehensive report with all visualizations and analysis"""
        Path(output_dir).mkdir(exist_ok=True)
        
        logger.info("Generating comprehensive BEA optimization report...")
        
        # 1. Run optimization analysis
        # Save results to temporary CSV for analysis (the analyzer expects a file path)
        temp_csv_path = f"{output_dir}/temp_results_for_analysis.csv"
        self.optimization_results.to_csv(temp_csv_path, index=False)
        analyze_optimization_results(temp_csv_path)
        
        # 2. Create geographic visualization
        self.create_geographic_visualization(f"{output_dir}/interactive_map.html")
        
        # 3. Create static visualizations
        self.create_static_visualizations(output_dir)
        
        # 4. Generate summary report
        self._generate_summary_report(output_dir)
        
        logger.info(f"Complete report generated in {output_dir}/")
        
    def _generate_summary_report(self, output_dir):
        """Generate text summary of optimization results"""
        with open(f"{output_dir}/optimization_summary.txt", 'w') as f:
            f.write("BEA SPECTRUM OPTIMIZATION SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall statistics
            total_stations = len(self.optimization_results)
            successful = self.optimization_results['optimized_start_freq_mhz'].notna().sum()
            failed = total_stations - successful
            
            f.write(f"Total Stations Processed: {total_stations}\n")
            f.write(f"Successfully Optimized: {successful} ({successful/total_stations*100:.1f}%)\n")
            f.write(f"Failed Allocations: {failed} ({failed/total_stations*100:.1f}%)\n\n")
            
            # BEA summary
            results_by_bea = self._aggregate_results_by_bea()
            f.write(f"Total BEA Regions with Stations: {len(results_by_bea)}\n")
            f.write(f"Average Stations per BEA: {results_by_bea['station_count'].mean():.1f}\n")
            f.write(f"Average Spectrum Efficiency: {results_by_bea['efficiency'].mean():.1f}%\n\n")
            
            # Top BEAs
            f.write("TOP 10 BEAs BY STATION COUNT:\n")
            f.write("-" * 40 + "\n")
            top_beas = results_by_bea.nlargest(10, 'station_count')
            for _, bea in top_beas.iterrows():
                f.write(f"{bea['bea_id']}: {bea['station_count']} stations, "
                       f"{bea['total_bandwidth']:.1f} MHz, "
                       f"{bea['efficiency']:.1f}% efficiency\n")
            
            # Method distribution
            f.write("\n\nOPTIMIZATION METHOD DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            method_counts = self.optimization_results['allocation_method'].value_counts()
            for method, count in method_counts.items():
                f.write(f"{method}: {count} ({count/total_stations*100:.1f}%)\n")


# Example usage function
def run_bea_optimization_demo():
    """Demonstration of the BEA spectrum optimization system"""
    
    # File paths - update these to your actual paths
    bea_geojson = "bea.geojson"
    bea_table = "example_bea_table.csv"
    
    # Initialize optimizer
    optimizer = BEASpectrumOptimizer(bea_geojson, bea_table)
    
    # Step 1: Load BEA data
    optimizer.load_bea_data()
    
    # Step 2: Prepare optimization input
    optimization_df = optimizer.prepare_optimization_input()
    
    # Step 3: Run optimization
    results = optimizer.run_optimization()
    
    # Step 4: Generate full report with visualizations
    optimizer.generate_full_report()
    
    return optimizer


if __name__ == "__main__":
    # Run the demonstration
    optimizer = run_bea_optimization_demo()
    print("\n✅ BEA spectrum optimization complete!")
    print("Check the 'bea_optimization_report' directory for results.")
