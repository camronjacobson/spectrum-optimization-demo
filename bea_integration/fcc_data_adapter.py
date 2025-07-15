# fcc_data_adapter.py
"""
FCC Data Adapter System
Handles real FCC data and transforms it for spectrum optimization
"""

import pandas as pd
import numpy as np
import requests
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import geopandas as gpd
from geopy.geocoders import Nominatim
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FCCDataAdapter:
    """
    Comprehensive adapter for real FCC data sources
    """
    
    def __init__(self, cache_dir: str = "fcc_data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.geocoder = Nominatim(user_agent="spectrum-optimizer")
        
        # Service-specific configurations
        self.service_configs = self._load_service_configs()
        
        # Coordinate cache to avoid repeated API calls
        self.coord_cache_file = self.cache_dir / "coordinate_cache.json"
        self.coord_cache = self._load_coordinate_cache()
        
    def _load_service_configs(self) -> Dict:
        """Load service-specific configurations"""
        return {
            'AM': {
                'bandwidth_mhz': 0.01,  # 10 kHz
                'frequency_range': (530, 1700),
                'api_service_code': 'AM'
            },
            'FM': {
                'bandwidth_mhz': 0.2,  # 200 kHz
                'frequency_range': (88, 108),
                'api_service_code': 'FM'
            },
            'TV': {
                'bandwidth_mhz': 6,  # 6 MHz
                'frequency_range': (54, 806),
                'api_service_code': 'TV'
            },
            'AWS': {
                'bandwidth_mhz': {'default': 10, 'pattern_match': {}},
                'frequency_range': (1710, 2155),
                'api_service_code': 'AW'
            },
            'LM': {  # Land Mobile
                'bandwidth_mhz': {'default': 0.025},  # 25 kHz default
                'frequency_range': (30, 3000),
                'api_service_code': 'LM'
            }
        }
    
    def process_fcc_data(self, 
                        input_file: str, 
                        service_type: Optional[str] = None,
                        bea_geojson_path: Optional[str] = None) -> pd.DataFrame:
        """
        Main entry point to process FCC data
        
        Parameters:
        -----------
        input_file : str
            Path to FCC data file (CSV format)
        service_type : str, optional
            Radio service type (AM, FM, TV, AWS, etc.)
        bea_geojson_path : str, optional
            Path to BEA shapes for geographic assignment
            
        Returns:
        --------
        pd.DataFrame
            Processed data ready for optimization
        """
        logger.info(f"Processing FCC data from {input_file}")
        
        # Load raw data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} records")
        
        # Detect service type if not provided
        if service_type is None:
            service_type = self._detect_service_type(df)
            logger.info(f"Detected service type: {service_type}")
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Add missing coordinates
        df = self._add_coordinates(df, service_type)
        
        # Add bandwidth information
        df = self._add_bandwidth(df, service_type)
        
        # Add antenna parameters
        df = self._add_antenna_params(df, service_type)
        
        # Assign to BEA regions if shapes provided
        if bea_geojson_path:
            df = self._assign_bea_regions(df, bea_geojson_path)
        else:
            df = self._assign_default_regions(df)
        
        # Add area type classification
        df = self._classify_area_types(df)
        
        # Validate and clean
        df = self._validate_and_clean(df)
        
        # Add optimization metadata
        df = self._add_optimization_metadata(df)
        
        logger.info(f"Processing complete. {len(df)} stations ready for optimization")
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to expected format"""
        column_mapping = {
            # Common variations of callsign
            'callsigns': 'station_id',
            'call_sign': 'station_id',
            'callsign': 'station_id',
            'call': 'station_id',
            
            # Licensee information
            'name': 'licensee',
            'entity_name': 'licensee',
            'licensee_name': 'licensee',
            
            # Service type
            'radio service': 'service_type',
            'radio_service': 'service_type',
            'service': 'service_type',
            
            # Location data (if present)
            'latitude': 'y_coord',
            'lat': 'y_coord',
            'longitude': 'x_coord',
            'lon': 'x_coord',
            'lng': 'x_coord',
            
            # Technical parameters
            'power': 'power_watts',
            'erp': 'power_watts',
            'frequency': 'assigned_freq_mhz',
            'freq': 'assigned_freq_mhz'
        }
        
        # Apply mapping
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Ensure critical columns exist
        if 'station_id' not in df.columns and 'callsigns' in df.columns:
            df['station_id'] = df['callsigns']
        
        return df
    
    def _detect_service_type(self, df: pd.DataFrame) -> str:
        """Auto-detect service type from data"""
        if 'service_type' in df.columns:
            # Get most common service type
            service_counts = df['service_type'].value_counts()
            return service_counts.index[0]
        
        # Try to detect from callsign patterns
        if 'station_id' in df.columns:
            sample_calls = df['station_id'].head(100)
            
            # AM stations often start with K or W followed by 3-4 letters
            if sample_calls.str.match(r'^[KW][A-Z]{2,3}$').sum() > 50:
                return 'AM'
            
            # FM stations often have -FM suffix
            if sample_calls.str.contains('-FM').sum() > 20:
                return 'FM'
            
            # TV stations often have -TV or -DT suffix
            if sample_calls.str.contains('-(TV|DT)').sum() > 10:
                return 'TV'
        
        return 'UNKNOWN'
    
    def _add_coordinates(self, df: pd.DataFrame, service_type: str) -> pd.DataFrame:
        """Add missing coordinate data"""
        logger.info("Adding coordinate data...")
        
        # Check if coordinates already exist
        has_coords = ('x_coord' in df.columns and 'y_coord' in df.columns and
                     df['x_coord'].notna().sum() > len(df) * 0.8)
        
        if has_coords:
            logger.info("Coordinates already present for most stations")
            return df
        
        # Initialize coordinate columns if missing
        if 'x_coord' not in df.columns:
            df['x_coord'] = np.nan
        if 'y_coord' not in df.columns:
            df['y_coord'] = np.nan
        
        # Try different methods to get coordinates
        methods = [
            self._get_coords_from_fcc_api,
            self._get_coords_from_address,
            self._get_coords_from_city_state,
            self._get_coords_from_callsign_pattern
        ]
        
        for idx, row in df.iterrows():
            if pd.notna(row.get('x_coord')) and pd.notna(row.get('y_coord')):
                continue
                
            for method in methods:
                coords = method(row, service_type)
                if coords:
                    df.loc[idx, 'x_coord'] = coords[0]
                    df.loc[idx, 'y_coord'] = coords[1]
                    break
            
            # Progress logging
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(df)} stations for coordinates")
        
        # Save coordinate cache
        self._save_coordinate_cache()
        
        # Fill remaining with regional defaults
        missing_coords = df['x_coord'].isna() | df['y_coord'].isna()
        if missing_coords.any():
            logger.warning(f"{missing_coords.sum()} stations still missing coordinates")
            df = self._fill_with_regional_coords(df)
        
        return df
    
    def _get_coords_from_fcc_api(self, row: pd.Series, service_type: str) -> Optional[Tuple[float, float]]:
        """Query FCC ULS API for coordinates"""
        station_id = row.get('station_id', '')
        
        # Check cache first
        if station_id in self.coord_cache:
            return self.coord_cache[station_id]
        
        # This is a placeholder - actual FCC API integration would go here
        # For now, return None to try other methods
        return None
    
    def _get_coords_from_address(self, row: pd.Series, service_type: str) -> Optional[Tuple[float, float]]:
        """Geocode from address if available"""
        address_parts = []
        
        for field in ['address', 'street', 'city', 'state', 'zip']:
            if field in row and pd.notna(row[field]):
                address_parts.append(str(row[field]))
        
        if not address_parts:
            return None
        
        address = ', '.join(address_parts)
        
        try:
            location = self.geocoder.geocode(address, timeout=5)
            if location:
                coords = (location.longitude, location.latitude)
                # Cache the result
                if 'station_id' in row:
                    self.coord_cache[row['station_id']] = coords
                return coords
        except Exception as e:
            logger.debug(f"Geocoding failed for {address}: {e}")
        
        return None
    
    def _get_coords_from_city_state(self, row: pd.Series, service_type: str) -> Optional[Tuple[float, float]]:
        """Get approximate coordinates from city/state"""
        # Extract city/state from licensee name or other fields
        if 'licensee' in row and pd.notna(row['licensee']):
            # Look for patterns like "Company, City, ST"
            match = re.search(r',\s*([^,]+),\s*([A-Z]{2})$', row['licensee'])
            if match:
                city, state = match.groups()
                return self._get_city_coords(city.strip(), state.strip())
        
        return None
    
    def _get_coords_from_callsign_pattern(self, row: pd.Series, service_type: str) -> Optional[Tuple[float, float]]:
        """Estimate coordinates from callsign patterns"""
        if 'station_id' not in row:
            return None
            
        callsign = row['station_id']
        
        # US callsign regions (rough approximation)
        if callsign.startswith('K'):
            # West of Mississippi
            return (-115.0, 40.0)  # Western US center
        elif callsign.startswith('W'):
            # East of Mississippi
            return (-80.0, 40.0)  # Eastern US center
        
        return None
    
    def _get_city_coords(self, city: str, state: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a city/state combination"""
        cache_key = f"{city},{state}"
        if cache_key in self.coord_cache:
            return self.coord_cache[cache_key]
        
        try:
            location = self.geocoder.geocode(f"{city}, {state}, USA", timeout=5)
            if location:
                coords = (location.longitude, location.latitude)
                self.coord_cache[cache_key] = coords
                return coords
        except:
            pass
        
        return None
    
    def _fill_with_regional_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing coordinates with regional approximations"""
        # State geographic centers
        state_centers = {
            'AL': (-86.8, 32.8), 'AK': (-152.0, 64.0), 'AZ': (-112.1, 34.3),
            'AR': (-92.4, 34.9), 'CA': (-119.4, 37.2), 'CO': (-105.5, 39.0),
            'CT': (-72.7, 41.6), 'DE': (-75.5, 39.0), 'FL': (-81.7, 28.6),
            'GA': (-83.4, 32.7), 'HI': (-157.8, 21.3), 'ID': (-114.7, 44.1),
            'IL': (-89.4, 40.0), 'IN': (-86.3, 39.9), 'IA': (-93.5, 42.0),
            'KS': (-98.4, 38.5), 'KY': (-85.3, 37.5), 'LA': (-92.0, 31.0),
            'ME': (-69.2, 45.3), 'MD': (-76.8, 39.0), 'MA': (-71.8, 42.3),
            'MI': (-84.7, 43.3), 'MN': (-94.3, 46.3), 'MS': (-89.7, 32.7),
            'MO': (-92.5, 38.4), 'MT': (-109.5, 47.0), 'NE': (-99.8, 41.5),
            'NV': (-116.6, 39.3), 'NH': (-71.6, 43.7), 'NJ': (-74.7, 40.2),
            'NM': (-106.1, 34.4), 'NY': (-75.5, 42.9), 'NC': (-79.4, 35.6),
            'ND': (-100.5, 47.5), 'OH': (-82.8, 40.2), 'OK': (-97.5, 35.5),
            'OR': (-120.5, 43.9), 'PA': (-77.8, 40.9), 'RI': (-71.5, 41.7),
            'SC': (-80.9, 33.9), 'SD': (-100.3, 44.4), 'TN': (-86.3, 35.8),
            'TX': (-98.5, 31.5), 'UT': (-111.9, 39.3), 'VT': (-72.7, 44.0),
            'VA': (-78.8, 37.5), 'WA': (-120.4, 47.4), 'WV': (-80.6, 38.6),
            'WI': (-89.6, 44.6), 'WY': (-107.6, 43.0)
        }
        
        # Try to extract state from various fields
        for idx, row in df.iterrows():
            if pd.notna(row.get('x_coord')) and pd.notna(row.get('y_coord')):
                continue
            
            # Look for state code
            state = None
            for field in ['state', 'licensee', 'address']:
                if field in row and pd.notna(row[field]):
                    # Extract 2-letter state code
                    match = re.search(r'\b([A-Z]{2})\b', str(row[field]))
                    if match and match.group(1) in state_centers:
                        state = match.group(1)
                        break
            
            if state:
                coords = state_centers[state]
                # Add some random offset to avoid all stations at same point
                offset_x = np.random.normal(0, 0.5)
                offset_y = np.random.normal(0, 0.5)
                df.loc[idx, 'x_coord'] = coords[0] + offset_x
                df.loc[idx, 'y_coord'] = coords[1] + offset_y
            else:
                # Default to US center with larger random offset
                df.loc[idx, 'x_coord'] = -98.5 + np.random.normal(0, 10)
                df.loc[idx, 'y_coord'] = 39.8 + np.random.normal(0, 5)
        
        return df
    
    def _add_bandwidth(self, df: pd.DataFrame, service_type: str) -> pd.DataFrame:
        """Add bandwidth information based on service type"""
        logger.info("Adding bandwidth information...")
        
        if 'bandwidth_mhz' in df.columns and df['bandwidth_mhz'].notna().all():
            return df
        
        if 'bandwidth_mhz' not in df.columns:
            df['bandwidth_mhz'] = np.nan
        
        # Get service configuration
        config = self.service_configs.get(service_type, {})
        
        if isinstance(config.get('bandwidth_mhz'), dict):
            # Complex bandwidth assignment (e.g., AWS)
            default_bw = config['bandwidth_mhz']['default']
            pattern_map = config['bandwidth_mhz'].get('pattern_match', {})
            
            for idx, row in df.iterrows():
                if pd.notna(df.loc[idx, 'bandwidth_mhz']):
                    continue
                    
                assigned = False
                # Check patterns
                for pattern, bw in pattern_map.items():
                    if pattern in str(row.get('licensee', '')):
                        df.loc[idx, 'bandwidth_mhz'] = bw
                        assigned = True
                        break
                
                if not assigned:
                    df.loc[idx, 'bandwidth_mhz'] = default_bw
        else:
            # Simple bandwidth assignment
            default_bw = config.get('bandwidth_mhz', 10)
            df.loc[df['bandwidth_mhz'].isna(), 'bandwidth_mhz'] = default_bw
        
        return df
    
    def _add_antenna_params(self, df: pd.DataFrame, service_type: str) -> pd.DataFrame:
        """Add antenna parameters based on service type"""
        logger.info("Adding antenna parameters...")
        
        # Initialize columns
        for col in ['azimuth_deg', 'elevation_deg', 'power_watts']:
            if col not in df.columns:
                df[col] = np.nan
        
        # Service-specific antenna patterns
        if service_type == 'AM':
            # AM is typically omnidirectional
            df.loc[df['azimuth_deg'].isna(), 'azimuth_deg'] = 0
            df.loc[df['elevation_deg'].isna(), 'elevation_deg'] = 0
        
        elif service_type == 'FM':
            # FM can be directional or omnidirectional
            # If no data, assume omnidirectional
            df.loc[df['azimuth_deg'].isna(), 'azimuth_deg'] = 0
            df.loc[df['elevation_deg'].isna(), 'elevation_deg'] = 0
        
        elif service_type == 'TV':
            # TV often has directional patterns
            # Assign random azimuths if missing
            mask = df['azimuth_deg'].isna()
            df.loc[mask, 'azimuth_deg'] = np.random.uniform(0, 360, mask.sum())
            df.loc[df['elevation_deg'].isna(), 'elevation_deg'] = 0
        
        elif service_type in ['AWS', 'CELLULAR']:
            # Cellular typically uses 3-sector sites
            mask = df['azimuth_deg'].isna()
            # Randomly assign to one of three sectors
            sectors = np.random.choice([0, 120, 240], size=mask.sum())
            df.loc[mask, 'azimuth_deg'] = sectors
            # Slight downtilt
            df.loc[df['elevation_deg'].isna(), 'elevation_deg'] = np.random.uniform(-10, -2, mask.sum())
        
        else:
            # Default: omnidirectional
            df.loc[df['azimuth_deg'].isna(), 'azimuth_deg'] = 0
            df.loc[df['elevation_deg'].isna(), 'elevation_deg'] = 0
        
        return df
    
    def _assign_bea_regions(self, df: pd.DataFrame, bea_geojson_path: str) -> pd.DataFrame:
        """Assign stations to BEA regions"""
        logger.info("Assigning stations to BEA regions...")
        
        # Load BEA shapes
        bea_shapes = gpd.read_file(bea_geojson_path)
        
        # Create GeoDataFrame from stations
        station_gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.x_coord, df.y_coord),
            crs='EPSG:4326'
        )
        
        # Ensure same CRS
        if bea_shapes.crs != station_gdf.crs:
            bea_shapes = bea_shapes.to_crs(station_gdf.crs)
        
        # Spatial join
        stations_with_bea = gpd.sjoin(
            station_gdf,
            bea_shapes,
            how='left',
            predicate='within'
        )
        
        # Extract BEA information
        df['bea_id'] = stations_with_bea.get('bea', 'Unknown')
        df['bea_name'] = stations_with_bea.get('Name', 'Unknown')
        
        # Handle unassigned stations
        unassigned = df['bea_id'] == 'Unknown'
        if unassigned.any():
            logger.warning(f"{unassigned.sum()} stations could not be assigned to BEA regions")
            # Find nearest BEA for unassigned stations
            df = self._assign_nearest_bea(df, bea_shapes, unassigned)
        
        # Create clusters within BEAs
        df = self._create_clusters(df)
        
        # Extract state from BEA name
        df['state'] = df['bea_name'].apply(self._extract_state_from_bea)
        
        return df
    
    def _assign_default_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign default regions when BEA shapes not available"""
        logger.info("Assigning default regions based on coordinates...")
        
        # Simple grid-based assignment
        df['bea_id'] = 'GRID_' + \
                       (df['x_coord'] // 5).astype(str) + '_' + \
                       (df['y_coord'] // 5).astype(str)
        df['bea_name'] = df['bea_id']
        
        # Create clusters
        df['cluster'] = df['bea_id'] + '_0'
        
        # Estimate state from coordinates
        df['state'] = df.apply(self._coord_to_state, axis=1)
        
        return df
    
    def _create_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sub-clusters within BEA regions"""
        # For now, simple approach - could be enhanced with actual clustering
        df['cluster'] = df['bea_id'].astype(str) + '_0'
        
        # For large BEAs, create multiple clusters
        bea_counts = df['bea_id'].value_counts()
        large_beas = bea_counts[bea_counts > 100].index
        
        for bea in large_beas:
            mask = df['bea_id'] == bea
            n_stations = mask.sum()
            n_clusters = min(5, n_stations // 50)  # Max 5 clusters per BEA
            
            if n_clusters > 1:
                # Simple spatial clustering
                bea_stations = df[mask]
                kmeans_labels = self._simple_spatial_clustering(
                    bea_stations[['x_coord', 'y_coord']].values,
                    n_clusters
                )
                df.loc[mask, 'cluster'] = bea + '_' + kmeans_labels.astype(str)
        
        return df
    
    def _classify_area_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify stations into urban/suburban/rural"""
        logger.info("Classifying area types...")
        
        if 'area_type' in df.columns and df['area_type'].notna().all():
            return df
        
        if 'area_type' not in df.columns:
            df['area_type'] = 'suburban'  # Default
        
        # Population density based classification would go here
        # For now, use simple heuristics
        
        # Major metropolitan BEAs
        urban_keywords = ['New York', 'Los Angeles', 'Chicago', 'Houston', 
                         'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego',
                         'Dallas', 'San Jose', 'Austin', 'Jacksonville']
        
        rural_keywords = ['Rural', 'Non-Metro', 'Micropolitan', 'County']
        
        for keyword in urban_keywords:
            mask = df['bea_name'].str.contains(keyword, case=False, na=False)
            df.loc[mask, 'area_type'] = 'urban'
        
        for keyword in rural_keywords:
            mask = df['bea_name'].str.contains(keyword, case=False, na=False)
            df.loc[mask, 'area_type'] = 'rural'
        
        # Station density based classification
        cluster_density = df.groupby('cluster').size()
        high_density_clusters = cluster_density[cluster_density > 50].index
        low_density_clusters = cluster_density[cluster_density < 10].index
        
        df.loc[df['cluster'].isin(high_density_clusters), 'area_type'] = 'urban'
        df.loc[df['cluster'].isin(low_density_clusters), 'area_type'] = 'rural'
        
        return df
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the processed data"""
        logger.info("Validating and cleaning data...")
        
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['station_id'], keep='first')
        
        # Remove invalid coordinates
        valid_coords = (
            (df['x_coord'] >= -180) & (df['x_coord'] <= 180) &
            (df['y_coord'] >= -90) & (df['y_coord'] <= 90)
        )
        df = df[valid_coords]
        
        # Remove invalid bandwidth
        df = df[df['bandwidth_mhz'] > 0]
        
        # Ensure required columns have no nulls
        required_cols = ['station_id', 'x_coord', 'y_coord', 'bandwidth_mhz',
                        'azimuth_deg', 'elevation_deg', 'cluster', 'state', 
                        'area_type']
        
        for col in required_cols:
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    logger.warning(f"Column {col} has {null_count} null values")
        
        final_count = len(df)
        if final_count < initial_count:
            logger.info(f"Removed {initial_count - final_count} invalid records")
        
        return df
    
    def _add_optimization_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata useful for optimization"""
        
        # Add processing timestamp
        df['processed_at'] = datetime.now().isoformat()
        
        # Add data quality score
        df['data_quality_score'] = 1.0
        
        # Reduce score for estimated data
        if 'coord_estimated' in df.columns:
            df.loc[df['coord_estimated'], 'data_quality_score'] -= 0.2
        
        # Sort by station ID for consistent ordering
        df = df.sort_values('station_id').reset_index(drop=True)
        
        return df
    
    def _extract_state_from_bea(self, bea_name: str) -> str:
        """Extract state code from BEA name"""
        if pd.isna(bea_name) or bea_name == 'Unknown':
            return 'XX'
        
        # Most BEA names end with ", STATE"
        parts = bea_name.split(', ')
        if len(parts) >= 2:
            potential_state = parts[-1].strip()
            if len(potential_state) == 2 and potential_state.isalpha():
                return potential_state.upper()
        
        return 'XX'
    
    def _coord_to_state(self, row: pd.Series) -> str:
        """Estimate state from coordinates"""
        # This is a simplified approach
        # In practice, you'd use a proper reverse geocoding service
        x, y = row['x_coord'], row['y_coord']
        
        if x < -100:  # Western states
            if y > 42:
                return 'WA' if x < -120 else 'MT'
            elif y > 35:
                return 'CA' if x < -115 else 'NV'
            else:
                return 'AZ' if y < 33 else 'NM'
        else:  # Eastern states
            if y > 42:
                return 'NY' if x > -75 else 'MI'
            elif y > 35:
                return 'NC' if x > -80 else 'TN'
            else:
                return 'FL' if x > -82 else 'TX'
    
    def _simple_spatial_clustering(self, coords: np.ndarray, n_clusters: int) -> np.ndarray:
        """Simple k-means style clustering"""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(coords)
    
    def _load_coordinate_cache(self) -> Dict:
        """Load coordinate cache from disk"""
        if self.coord_cache_file.exists():
            try:
                with open(self.coord_cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_coordinate_cache(self):
        """Save coordinate cache to disk"""
        try:
            with open(self.coord_cache_file, 'w') as f:
                json.dump(self.coord_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save coordinate cache: {e}")


def process_fcc_file(input_file: str, 
                    output_file: str,
                    service_type: Optional[str] = None,
                    bea_geojson: Optional[str] = None) -> bool:
    """
    Convenience function to process an FCC file
    
    Parameters:
    -----------
    input_file : str
        Path to input FCC data file
    output_file : str
        Path to save processed data
    service_type : str, optional
        Radio service type
    bea_geojson : str, optional
        Path to BEA shapes file
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        adapter = FCCDataAdapter()
        processed_df = adapter.process_fcc_data(input_file, service_type, bea_geojson)
        processed_df.to_csv(output_file, index=False)
        
        logger.info(f"Successfully processed {len(processed_df)} stations")
        logger.info(f"Output saved to {output_file}")
        
        # Print summary statistics
        print("\nProcessing Summary:")
        print(f"Total stations: {len(processed_df)}")
        print(f"BEA regions: {processed_df['bea_id'].nunique()}")
        print(f"Clusters: {processed_df['cluster'].nunique()}")
        print(f"States: {processed_df['state'].nunique()}")
        print("\nArea type distribution:")
        print(processed_df['area_type'].value_counts())
        print("\nBandwidth statistics:")
        print(processed_df['bandwidth_mhz'].describe())
        
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Example usage
    success = process_fcc_file(
        input_file="fcc_data.csv",
        output_file="processed_fcc_data.csv",
        service_type="AWS",  # or None for auto-detect
        bea_geojson="bea.geojson"
    )
    
    if success:
        print("\n✅ FCC data processing complete!")
        print("You can now use 'processed_fcc_data.csv' with your spectrum optimizer")
    else:
        print("\n❌ FCC data processing failed!")
