# bea_market_data_adapter.py
"""
BEA Market Data Adapter for Spectrum Optimization
Handles both market-level and station-level data with automatic detection
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import logging
import re
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BEAMarketDataAdapter:
    """
    Adapter to convert market-level BEA data to station-level data for optimization
    """
    
    def __init__(self, bea_geojson_path: str):
        """
        Initialize the adapter with BEA shapes
        
        Parameters:
        -----------
        bea_geojson_path : str
            Path to BEA GeoJSON file with polygon boundaries
        """
        self.bea_geojson_path = bea_geojson_path
        self.bea_shapes = None
        self.bea_mapping = None
        self._load_bea_shapes()
        
    def _load_bea_shapes(self):
        """Load and prepare BEA shapes"""
        logger.info(f"Loading BEA shapes from {self.bea_geojson_path}")
        
        self.bea_shapes = gpd.read_file(self.bea_geojson_path)
        
        # Ensure we have the required fields
        if 'bea' not in self.bea_shapes.columns or 'Name' not in self.bea_shapes.columns:
            raise ValueError("BEA GeoJSON must have 'bea' and 'Name' fields")
        
        # Create mapping dictionary
        self.bea_mapping = {}
        for idx, row in self.bea_shapes.iterrows():
            bea_code = int(row['bea'])
            self.bea_mapping[bea_code] = {
                'name': row['Name'],
                'geometry': row['geometry']
            }
        
        logger.info(f"Loaded {len(self.bea_mapping)} BEA regions")
        
    def process_data(self, 
                     input_file: str, 
                     output_file: str,
                     bea_mapping_file: Optional[str] = None,
                     station_density: str = 'auto') -> pd.DataFrame:
        """
        Process market or station level data
        
        Parameters:
        -----------
        input_file : str
            Path to input CSV file
        output_file : str
            Path to save processed station-level data
        bea_mapping_file : str, optional
            Path to BEA code/name mapping CSV if needed
        station_density : str
            'auto', 'low', 'medium', 'high' - controls station distribution
            
        Returns:
        --------
        pd.DataFrame
            Processed station-level data ready for optimization
        """
        # Load input data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} records from {input_file}")
        
        # Detect data type
        data_mode = self._detect_data_mode(df)
        logger.info(f"Detected data mode: {data_mode}")
        
        if data_mode == "station_level":
            result_df = self._process_station_level_data(df)
        else:
            result_df = self._process_market_level_data(df, bea_mapping_file, station_density)
        
        # Validate and clean
        result_df = self._validate_and_clean(result_df)
        
        # Sort by station_id for consistency
        result_df = result_df.sort_values('station_id').reset_index(drop=True)
        
        # Save results
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(result_df)} stations to {output_file}")
        
        # Print summary
        self._print_summary(result_df)
        
        return result_df
    
    def _detect_data_mode(self, df: pd.DataFrame) -> str:
        """Detect whether data is market-level or station-level"""
        # Check for station coordinates
        if all(col in df.columns for col in ['x_coord', 'y_coord']):
            if df['x_coord'].notna().sum() > len(df) * 0.5:
                return "station_level"
        
        # Check for market column
        if 'market' in df.columns:
            # Check if market contains BEA codes
            if df['market'].str.contains('BEA\\d{3}', na=False).any():
                return "market_level"
        
        # Default to market level
        return "market_level"
    
    def _process_station_level_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process station-level data with coordinates"""
        logger.info("Processing station-level data...")
        
        result_df = df.copy()
        
        # Ensure required columns
        if 'station_id' not in result_df.columns:
            if 'callsign' in result_df.columns:
                result_df['station_id'] = result_df['callsign']
            else:
                result_df['station_id'] = ['STATION_' + str(i) for i in range(len(result_df))]
        
        # Assign BEA regions if not present
        if 'bea_id' not in result_df.columns:
            result_df = self._assign_stations_to_bea(result_df)
        
        # Add required fields for optimization
        result_df = self._add_optimization_fields(result_df)
        
        return result_df
    
    def _process_market_level_data(self, 
                                   df: pd.DataFrame, 
                                   bea_mapping_file: Optional[str],
                                   station_density: str) -> pd.DataFrame:
        """Process market-level data by creating synthetic stations"""
        logger.info("Processing market-level data...")
        
        # Parse BEA codes from market column
        df = self._parse_market_column(df)
        
        # Load BEA mapping if provided
        if bea_mapping_file:
            bea_name_mapping = pd.read_csv(bea_mapping_file)
            # Could use this for validation
        
        # Group by BEA to count licenses per market
        licenses_per_bea = df.groupby('bea_code').size().to_dict()
        
        # Generate stations for each BEA
        all_stations = []
        station_counter = 1
        
        for bea_code, num_licenses in licenses_per_bea.items():
            if bea_code not in self.bea_mapping:
                logger.warning(f"BEA code {bea_code} not found in GeoJSON, skipping...")
                continue
            
            # Get licenses for this BEA
            bea_licenses = df[df['bea_code'] == bea_code]
            
            # Generate station locations
            stations = self._generate_stations_for_bea(
                bea_code, 
                num_licenses, 
                bea_licenses,
                station_density,
                station_counter
            )
            
            all_stations.extend(stations)
            station_counter += len(stations)
        
        # Create DataFrame
        result_df = pd.DataFrame(all_stations)
        
        return result_df
    
    def _parse_market_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse BEA codes from market column"""
        logger.info("Parsing market column...")
        
        def extract_bea_code(market_str):
            """Extract numeric BEA code from market string"""
            if pd.isna(market_str):
                return None
            
            # Clean whitespace and non-breaking spaces
            market_str = str(market_str).replace('\xa0', ' ').strip()
            
            # Extract BEA code
            match = re.search(r'BEA(\d{3})', market_str)
            if match:
                return int(match.group(1))
            return None
        
        def extract_bea_name(market_str):
            """Extract BEA name from market string"""
            if pd.isna(market_str):
                return None
            
            # Clean whitespace
            market_str = str(market_str).replace('\xa0', ' ').strip()
            
            # Split on dash and take the part after BEA code
            parts = market_str.split('-', 1)
            if len(parts) > 1:
                return parts[1].strip()
            return None
        
        df['bea_code'] = df['market'].apply(extract_bea_code)
        df['bea_name'] = df['market'].apply(extract_bea_name)
        
        # Log any parsing failures
        failed = df['bea_code'].isna()
        if failed.any():
            logger.warning(f"Failed to parse {failed.sum()} market entries")
        
        return df
    
    def _generate_stations_for_bea(self,
                                   bea_code: int,
                                   num_stations: int,
                                   licenses_df: pd.DataFrame,
                                   density: str,
                                   start_counter: int) -> List[Dict]:
        """Generate synthetic station locations within a BEA polygon"""
        geometry = self.bea_mapping[bea_code]['geometry']
        bea_name = self.bea_mapping[bea_code]['name']
        
        logger.info(f"Generating {num_stations} stations for BEA{bea_code:03d} - {bea_name}")
        
        # Get placement points
        if density == 'low' or num_stations == 1:
            # Just use centroid
            points = [geometry.centroid]
        else:
            # Generate distributed points
            points = self._generate_distributed_points(geometry, num_stations, density)
        
        # Create station records
        stations = []
        licenses_list = licenses_df.to_dict('records')
        
        for i, point in enumerate(points):
            if i >= len(licenses_list):
                # More points than licenses - shouldn't happen but just in case
                break
                
            license_data = licenses_list[i]
            
            station = {
                'station_id': f'BEA{bea_code:03d}_S{start_counter + i:04d}',
                'x_coord': point.x,
                'y_coord': point.y,
                'bea_id': bea_code,
                'bea_name': bea_name,
                'cluster': f'{bea_code}_0',  # Default cluster
                'state': self._extract_state_from_bea_name(bea_name),
                'area_type': self._determine_area_type(bea_name, num_stations),
                
                # License data
                'licensee': license_data.get('licensee', 'Unknown'),
                'bandwidth_mhz': license_data.get('bandwidth_mhz', 10),
                'lower_freq': license_data.get('lower freq', 2000),
                'upper_freq': license_data.get('upper freq', 2010),
                'channel_block': license_data.get('channel block', 'A'),
                
                # Technical parameters
                'azimuth_deg': np.random.uniform(0, 360),  # Random for now
                'elevation_deg': 0,
                'power_watts': 100,  # Default
                
                # Metadata
                'original_market': license_data.get('market', ''),
                'grant_date': license_data.get('grant date', ''),
                'effective_date': license_data.get('effective date', '')
            }
            
            stations.append(station)
        
        return stations
    
    def _generate_distributed_points(self, 
                                    geometry: Union[Polygon, gpd.GeoSeries],
                                    num_points: int,
                                    density: str) -> List[Point]:
        """Generate distributed points within a polygon"""
        points = []
        
        # Get bounds
        minx, miny, maxx, maxy = geometry.bounds
        
        # Determine grid size based on density
        if density == 'high' or num_points > 20:
            grid_size = int(np.ceil(np.sqrt(num_points * 1.5)))
        elif density == 'medium' or num_points > 10:
            grid_size = int(np.ceil(np.sqrt(num_points * 1.2)))
        else:
            grid_size = int(np.ceil(np.sqrt(num_points)))
        
        # Generate grid points
        x_step = (maxx - minx) / (grid_size + 1)
        y_step = (maxy - miny) / (grid_size + 1)
        
        grid_points = []
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                x = minx + i * x_step
                y = miny + j * y_step
                point = Point(x, y)
                
                # Check if point is within polygon
                if geometry.contains(point):
                    grid_points.append(point)
        
        # If we have enough grid points, sample from them
        if len(grid_points) >= num_points:
            indices = np.random.choice(len(grid_points), num_points, replace=False)
            points = [grid_points[i] for i in indices]
        else:
            # Use grid points and add random points
            points = grid_points
            
            # Add random points until we have enough
            max_attempts = num_points * 10
            attempts = 0
            
            while len(points) < num_points and attempts < max_attempts:
                x = np.random.uniform(minx, maxx)
                y = np.random.uniform(miny, maxy)
                point = Point(x, y)
                
                if geometry.contains(point):
                    # Check minimum distance to existing points
                    min_dist = float('inf')
                    for existing in points:
                        dist = point.distance(existing)
                        min_dist = min(min_dist, dist)
                    
                    # Add if far enough from others
                    if min_dist > (maxx - minx) / (grid_size * 2):
                        points.append(point)
                
                attempts += 1
        
        return points[:num_points]
    
    def _assign_stations_to_bea(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign stations to BEA regions using spatial join"""
        logger.info("Assigning stations to BEA regions...")
        
        # Create GeoDataFrame from stations
        station_gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.x_coord, df.y_coord),
            crs='EPSG:4326'
        )
        
        # Ensure same CRS
        if self.bea_shapes.crs != station_gdf.crs:
            self.bea_shapes = self.bea_shapes.to_crs(station_gdf.crs)
        
        # Spatial join
        joined = gpd.sjoin(station_gdf, self.bea_shapes, how='left', predicate='within')
        
        # Extract BEA info
        df['bea_id'] = joined['bea']
        df['bea_name'] = joined['Name']
        
        # Handle unassigned stations (find nearest)
        unassigned = df['bea_id'].isna()
        if unassigned.any():
            logger.warning(f"{unassigned.sum()} stations not within any BEA, assigning to nearest...")
            df = self._assign_nearest_bea(df, unassigned)
        
        return df
    
    def _assign_nearest_bea(self, df: pd.DataFrame, unassigned_mask: pd.Series) -> pd.DataFrame:
        """Assign unassigned stations to nearest BEA"""
        unassigned_points = gpd.GeoDataFrame(
            df[unassigned_mask],
            geometry=gpd.points_from_xy(
                df[unassigned_mask].x_coord,
                df[unassigned_mask].y_coord
            ),
            crs='EPSG:4326'
        )
        
        for idx, point in unassigned_points.iterrows():
            # Find nearest BEA
            distances = self.bea_shapes.geometry.distance(point.geometry)
            nearest_idx = distances.idxmin()
            nearest_bea = self.bea_shapes.loc[nearest_idx]
            
            df.loc[idx, 'bea_id'] = nearest_bea['bea']
            df.loc[idx, 'bea_name'] = nearest_bea['Name']
        
        return df
    
    def _add_optimization_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fields required for optimization"""
        # Add cluster if missing
        if 'cluster' not in df.columns:
            df['cluster'] = df['bea_id'].astype(str) + '_0'
        
        # Add state if missing
        if 'state' not in df.columns:
            df['state'] = df['bea_name'].apply(self._extract_state_from_bea_name)
        
        # Add area type if missing
        if 'area_type' not in df.columns:
            # Estimate based on station density
            station_counts = df['bea_id'].value_counts()
            df['area_type'] = df['bea_id'].apply(
                lambda x: self._determine_area_type_by_density(station_counts.get(x, 1))
            )
        
        # Add technical fields if missing
        if 'azimuth_deg' not in df.columns:
            df['azimuth_deg'] = 0
        if 'elevation_deg' not in df.columns:
            df['elevation_deg'] = 0
        if 'bandwidth_mhz' not in df.columns:
            df['bandwidth_mhz'] = 10  # Default
        
        return df
    
    def _extract_state_from_bea_name(self, bea_name: str) -> str:
        """Extract state code from BEA name"""
        if pd.isna(bea_name):
            return 'XX'
        
        # Look for state abbreviations
        states = re.findall(r'\b[A-Z]{2}\b', bea_name)
        
        if states:
            # Return the first valid state code
            return states[0]
        
        return 'XX'
    
    def _determine_area_type(self, bea_name: str, num_stations: int) -> str:
        """Determine area type based on BEA name and station count"""
        if pd.isna(bea_name):
            return 'suburban'
        
        # Major urban areas
        urban_keywords = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                         'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Miami',
                         'Atlanta', 'Washington', 'Boston', 'San Francisco', 'Seattle']
        
        for keyword in urban_keywords:
            if keyword.lower() in bea_name.lower():
                return 'urban'
        
        # Rural indicators
        if num_stations < 5:
            return 'rural'
        elif num_stations < 15:
            return 'suburban'
        else:
            return 'urban'
    
    def _determine_area_type_by_density(self, station_count: int) -> str:
        """Determine area type by station density"""
        if station_count >= 30:
            return 'urban'
        elif station_count >= 10:
            return 'suburban'
        else:
            return 'rural'
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the data"""
        logger.info("Validating and cleaning data...")
        
        initial_count = len(df)
        
        # Remove invalid coordinates
        valid_coords = (
            (df['x_coord'] >= -180) & (df['x_coord'] <= 180) &
            (df['y_coord'] >= -90) & (df['y_coord'] <= 90) &
            df['x_coord'].notna() & df['y_coord'].notna()
        )
        df = df[valid_coords]
        
        # Ensure positive bandwidth
        if 'bandwidth_mhz' in df.columns:
            df = df[df['bandwidth_mhz'] > 0]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['station_id'], keep='first')
        
        final_count = len(df)
        if final_count < initial_count:
            logger.info(f"Removed {initial_count - final_count} invalid records")
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"\nTotal stations generated: {len(df)}")
        print(f"BEA regions: {df['bea_id'].nunique()}")
        print(f"States: {df['state'].nunique()}")
        
        print("\nStations per BEA (top 10):")
        bea_counts = df.groupby(['bea_id', 'bea_name']).size().sort_values(ascending=False)
        for (bea_id, bea_name), count in bea_counts.head(10).items():
            print(f"  BEA{int(bea_id):03d} - {bea_name}: {count} stations")
        
        print("\nArea type distribution:")
        area_dist = df['area_type'].value_counts()
        for area_type, count in area_dist.items():
            print(f"  {area_type}: {count} ({count/len(df)*100:.1f}%)")
        
        print("\nFrequency range:")
        if 'lower_freq' in df.columns and 'upper_freq' in df.columns:
            print(f"  Min: {df['lower_freq'].min()} MHz")
            print(f"  Max: {df['upper_freq'].max()} MHz")
        
        print("\nTotal bandwidth:")
        if 'bandwidth_mhz' in df.columns:
            print(f"  {df['bandwidth_mhz'].sum():,.1f} MHz")
        
        print("="*60)


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BEA Market Data Adapter')
    parser.add_argument('input_file', help='Input CSV file with market or station data')
    parser.add_argument('output_file', help='Output CSV file for optimization')
    parser.add_argument('--bea-geojson', required=True, help='BEA GeoJSON file')
    parser.add_argument('--bea-mapping', help='BEA code/name mapping CSV')
    parser.add_argument('--density', choices=['auto', 'low', 'medium', 'high'],
                       default='auto', help='Station distribution density')
    
    args = parser.parse_args()
    
    # Process the data
    adapter = BEAMarketDataAdapter(args.bea_geojson)
    result = adapter.process_data(
        input_file=args.input_file,
        output_file=args.output_file,
        bea_mapping_file=args.bea_mapping,
        station_density=args.density
    )
    
    print(f"\n✅ Processing complete!")
    print(f"Output saved to: {args.output_file}")
    print(f"Ready for spectrum optimization")


if __name__ == "__main__":
    main()
