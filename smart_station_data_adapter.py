# smart_station_data_adapter.py
"""
Smart Station Data Adapter
Automatically detects and handles various station data formats
Integrates with the enhanced BEA optimization pipeline
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartStationDataAdapter:
    """
    Intelligent adapter that handles various station data formats
    including pre-processed, partially processed, and raw data
    """
    
    # Define expected columns for fully processed data
    REQUIRED_COLUMNS = {
        'station_id', 'x_coord', 'y_coord', 'bandwidth_mhz',
        'azimuth_deg', 'cluster', 'state', 'area_type'
    }
    
    OPTIONAL_COLUMNS = {
        'elevation_deg', 'licensee', 'city', 'power_watts',
        'bea_id', 'bea_name', 'service_type', 'frequency_mhz'
    }
    
    # Column mapping patterns for common formats
    COLUMN_MAPPINGS = {
        # Standard variations
        'callsign': 'station_id',
        'call_sign': 'station_id',
        'callsigns': 'station_id',
        'latitude': 'y_coord',
        'lat': 'y_coord',
        'longitude': 'x_coord',
        'lon': 'x_coord',
        'lng': 'x_coord',
        'bandwidth': 'bandwidth_mhz',
        'bw': 'bandwidth_mhz',
        'azimuth': 'azimuth_deg',
        'bearing': 'azimuth_deg',
        'elevation': 'elevation_deg',
        'tilt': 'elevation_deg',
        'owner': 'licensee',
        'operator': 'licensee',
        'location': 'city',
        'municipality': 'city',
        'power': 'power_watts',
        'erp': 'power_watts'
    }
    
    def __init__(self, bea_geojson_path: Optional[str] = None):
        """
        Initialize the adapter
        
        Parameters:
        -----------
        bea_geojson_path : str, optional
            Path to BEA GeoJSON file for geographic assignment
        """
        self.bea_geojson_path = bea_geojson_path
        self.bea_shapes = None
        if bea_geojson_path and Path(bea_geojson_path).exists():
            self.bea_shapes = gpd.read_file(bea_geojson_path)
            logger.info(f"Loaded {len(self.bea_shapes)} BEA regions")
    
    def process(self, input_file: str, service_type: Optional[str] = None) -> pd.DataFrame:
        """
        Main processing function that handles any station data format
        
        Parameters:
        -----------
        input_file : str
            Path to input CSV file
        service_type : str, optional
            Service type (AM, FM, TV, etc.) - will auto-detect if not provided
            
        Returns:
        --------
        pd.DataFrame
            Processed data ready for optimization
        """
        logger.info(f"Processing station data: {input_file}")
        
        # Load data
        df = pd.read_csv(input_file)
        original_columns = set(df.columns)
        logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")
        
        # Detect data completeness level
        completeness_level = self._detect_completeness_level(df)
        logger.info(f"Data completeness level: {completeness_level}")
        
        # Process based on completeness level
        if completeness_level == "fully_processed":
            logger.info("Data is already fully processed")
            df = self._validate_and_enhance_processed_data(df, service_type)
        
        elif completeness_level == "partially_processed":
            logger.info("Data is partially processed, filling missing components")
            df = self._complete_partial_data(df, service_type)
        
        else:  # raw_data
            logger.info("Data appears to be raw, performing full processing")
            df = self._process_raw_data(df, service_type)
        
        # Final validation
        df = self._final_validation(df)
        
        # Add metadata
        df['data_source_format'] = completeness_level
        df['processing_version'] = '2.0'
        
        logger.info(f"Processing complete: {len(df)} stations ready for optimization")
        
        return df
    
    def _detect_completeness_level(self, df: pd.DataFrame) -> str:
        """
        Detect how complete/processed the input data is
        
        Returns:
        --------
        str
            'fully_processed', 'partially_processed', or 'raw_data'
        """
        # Normalize column names for checking
        normalized_cols = set()
        for col in df.columns:
            normalized = col.lower().replace('_', '').replace(' ', '')
            normalized_cols.add(normalized)
        
        # Check for required columns (with normalization)
        required_found = 0
        for req_col in self.REQUIRED_COLUMNS:
            normalized_req = req_col.lower().replace('_', '')
            if normalized_req in normalized_cols or req_col in df.columns:
                required_found += 1
        
        # Determine completeness level
        completeness_ratio = required_found / len(self.REQUIRED_COLUMNS)
        
        if completeness_ratio >= 0.9:  # 90% or more required columns
            return "fully_processed"
        elif completeness_ratio >= 0.5:  # 50% or more
            return "partially_processed"
        else:
            return "raw_data"
    
    def _validate_and_enhance_processed_data(self, 
                                            df: pd.DataFrame, 
                                            service_type: Optional[str]) -> pd.DataFrame:
        """
        Validate fully processed data and add any missing enhancements
        """
        # Apply any necessary column mappings
        df = self._apply_column_mappings(df)
        
        # Add BEA assignments if missing
        if 'bea_id' not in df.columns and self.bea_shapes is not None:
            logger.info("Adding BEA assignments...")
            df = self._assign_bea_regions(df)
        
        # Add service type if missing
        if 'service_type' not in df.columns and service_type:
            df['service_type'] = service_type
        elif 'service_type' not in df.columns:
            df['service_type'] = self._infer_service_type(df)
        
        # Validate data quality
        df = self._validate_coordinates(df)
        df = self._validate_technical_params(df)
        
        return df
    
    def _complete_partial_data(self, 
                              df: pd.DataFrame, 
                              service_type: Optional[str]) -> pd.DataFrame:
        """
        Complete partially processed data by filling in missing components
        """
        # Apply column mappings first
        df = self._apply_column_mappings(df)
        
        # Fill missing coordinates if needed
        if 'x_coord' not in df.columns or df['x_coord'].isna().any():
            df = self._add_missing_coordinates(df)
        
        # Add missing technical parameters
        if 'bandwidth_mhz' not in df.columns or df['bandwidth_mhz'].isna().any():
            df = self._add_bandwidth(df, service_type)
        
        if 'azimuth_deg' not in df.columns:
            df = self._add_antenna_params(df, service_type)
        
        # Add clustering if missing
        if 'cluster' not in df.columns:
            df = self._create_clusters(df)
        
        # Add area types if missing
        if 'area_type' not in df.columns:
            df = self._classify_area_types(df)
        
        # Add BEA assignments
        if self.bea_shapes is not None:
            df = self._assign_bea_regions(df)
        
        return df
    
    def _process_raw_data(self, 
                         df: pd.DataFrame, 
                         service_type: Optional[str]) -> pd.DataFrame:
        """
        Process raw data from scratch
        """
        # This would implement full processing logic
        # For now, we'll raise an exception to use the existing FCC adapter
        raise NotImplementedError(
            "Raw data processing should use FCCDataAdapter. "
            "This adapter is optimized for pre-processed or partially processed data."
        )
    
    def _apply_column_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard column mappings"""
        for old_name, new_name in self.COLUMN_MAPPINGS.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
                logger.info(f"Mapped column: {old_name} -> {new_name}")
        
        return df
    
    def _assign_bea_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign stations to BEA regions using spatial join"""
        logger.info("Assigning stations to BEA regions...")
        
        # Create GeoDataFrame
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
        df['bea_id'] = joined.get('bea', 'Unknown')
        df['bea_name'] = joined.get('Name', 'Unknown')
        
        # Handle unassigned stations
        unassigned = df['bea_id'] == 'Unknown'
        if unassigned.any():
            logger.warning(f"{unassigned.sum()} stations outside BEA boundaries")
            df = self._assign_nearest_bea(df, unassigned)
        
        return df
    
    def _assign_nearest_bea(self, df: pd.DataFrame, unassigned_mask: pd.Series) -> pd.DataFrame:
        """Assign unassigned stations to nearest BEA"""
        unassigned_gdf = gpd.GeoDataFrame(
            df[unassigned_mask],
            geometry=gpd.points_from_xy(
                df[unassigned_mask].x_coord,
                df[unassigned_mask].y_coord
            ),
            crs='EPSG:4326'
        )
        
        for idx in unassigned_gdf.index:
            point = unassigned_gdf.loc[idx, 'geometry']
            distances = self.bea_shapes.geometry.distance(point)
            nearest_idx = distances.idxmin()
            
            df.loc[idx, 'bea_id'] = self.bea_shapes.loc[nearest_idx, 'bea']
            df.loc[idx, 'bea_name'] = self.bea_shapes.loc[nearest_idx, 'Name']
        
        return df
    
    def _add_bandwidth(self, df: pd.DataFrame, service_type: Optional[str]) -> pd.DataFrame:
        """Add bandwidth based on service type"""
        if 'bandwidth_mhz' not in df.columns:
            df['bandwidth_mhz'] = np.nan
        
        # Service-specific defaults
        bandwidth_defaults = {
            'AM': 0.01,  # 10 kHz
            'FM': 0.2,   # 200 kHz
            'TV': 6.0,   # 6 MHz
            'CELLULAR': 10.0,  # 10 MHz
            'AWS': 10.0  # 10 MHz
        }
        
        if service_type and service_type.upper() in bandwidth_defaults:
            default_bw = bandwidth_defaults[service_type.upper()]
            df.loc[df['bandwidth_mhz'].isna(), 'bandwidth_mhz'] = default_bw
        else:
            # Default to 10 MHz for unknown services
            df.loc[df['bandwidth_mhz'].isna(), 'bandwidth_mhz'] = 10.0
        
        return df
    
    def _add_antenna_params(self, df: pd.DataFrame, service_type: Optional[str]) -> pd.DataFrame:
        """Add antenna parameters if missing"""
        if 'azimuth_deg' not in df.columns:
            # Default to omnidirectional
            df['azimuth_deg'] = 0
        
        if 'elevation_deg' not in df.columns:
            df['elevation_deg'] = 0
        
        # Service-specific patterns
        if service_type and service_type.upper() in ['CELLULAR', 'AWS']:
            # Cellular typically uses 3-sector sites
            mask = df['azimuth_deg'] == 0
            if mask.any():
                sectors = np.random.choice([0, 120, 240], size=mask.sum())
                df.loc[mask, 'azimuth_deg'] = sectors
        
        return df
    
    def _create_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create clusters if missing"""
        if 'bea_id' in df.columns:
            # Use BEA-based clustering
            df['cluster'] = df['bea_id'].astype(str) + '_0'
            
            # For large BEAs, create sub-clusters
            bea_counts = df['bea_id'].value_counts()
            for bea_id in bea_counts[bea_counts > 50].index:
                mask = df['bea_id'] == bea_id
                # Simple spatial clustering within BEA
                df.loc[mask, 'cluster'] = self._spatial_clustering(
                    df[mask], 
                    prefix=f"{bea_id}_"
                )
        else:
            # Grid-based clustering
            df['cluster'] = 'GRID_' + \
                           (df['x_coord'] // 2).astype(str) + '_' + \
                           (df['y_coord'] // 2).astype(str)
        
        return df
    
    def _spatial_clustering(self, df_subset: pd.DataFrame, prefix: str) -> pd.Series:
        """Simple k-means clustering for subset of data"""
        from sklearn.cluster import KMeans
        
        n_clusters = min(5, len(df_subset) // 20)  # Max 5 clusters, min 20 stations per cluster
        if n_clusters <= 1:
            return prefix + '0'
        
        coords = df_subset[['x_coord', 'y_coord']].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(coords)
        
        return prefix + labels.astype(str)
    
    def _classify_area_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify stations into urban/suburban/rural"""
        if 'area_type' not in df.columns:
            # Default classification based on density
            if 'cluster' in df.columns:
                cluster_counts = df['cluster'].value_counts()
                
                df['area_type'] = df['cluster'].map(
                    lambda x: 'urban' if cluster_counts.get(x, 0) > 30
                    else 'rural' if cluster_counts.get(x, 0) < 10
                    else 'suburban'
                )
            else:
                df['area_type'] = 'suburban'  # Default
        
        return df
    
    def _infer_service_type(self, df: pd.DataFrame) -> str:
        """Infer service type from data characteristics"""
        # Check bandwidth
        if 'bandwidth_mhz' in df.columns:
            avg_bw = df['bandwidth_mhz'].mean()
            if avg_bw < 0.05:
                return 'AM'
            elif avg_bw < 0.5:
                return 'FM'
            elif avg_bw > 5:
                return 'TV'
        
        # Check station ID patterns
        if 'station_id' in df.columns:
            sample = df['station_id'].head(20)
            if sample.str.contains('-FM', na=False).any():
                return 'FM'
            elif sample.str.contains('-TV|-DT', na=False).any():
                return 'TV'
            elif sample.str.match(r'^[KW][A-Z]{2,3}$', na=False).sum() > 10:
                return 'AM'
        
        return 'UNKNOWN'
    
    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix coordinate issues"""
        # Check bounds
        invalid_coords = (
            (df['x_coord'] < -180) | (df['x_coord'] > 180) |
            (df['y_coord'] < -90) | (df['y_coord'] > 90)
        )
        
        if invalid_coords.any():
            logger.warning(f"Found {invalid_coords.sum()} invalid coordinates")
            # Set to NaN for now
            df.loc[invalid_coords, ['x_coord', 'y_coord']] = np.nan
        
        return df
    
    def _validate_technical_params(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate technical parameters"""
        # Bandwidth should be positive
        if 'bandwidth_mhz' in df.columns:
            invalid_bw = df['bandwidth_mhz'] <= 0
            if invalid_bw.any():
                logger.warning(f"Found {invalid_bw.sum()} invalid bandwidth values")
                df.loc[invalid_bw, 'bandwidth_mhz'] = 10.0  # Default
        
        # Azimuth should be 0-360
        if 'azimuth_deg' in df.columns:
            df['azimuth_deg'] = df['azimuth_deg'] % 360
        
        return df
    
    def _add_missing_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing coordinates - simplified version"""
        logger.warning("Adding missing coordinates - using regional approximations")
        
        # For missing coordinates, use state centers or BEA centers
        if 'state' in df.columns:
            # Use state centers (simplified)
            state_centers = {
                'CA': (-119.4, 37.2), 'TX': (-98.5, 31.5),
                'NY': (-75.5, 42.9), 'FL': (-81.7, 28.6),
                # Add more as needed
            }
            
            for state, (x, y) in state_centers.items():
                mask = (df['state'] == state) & df['x_coord'].isna()
                if mask.any():
                    # Add some randomness
                    df.loc[mask, 'x_coord'] = x + np.random.normal(0, 0.5, mask.sum())
                    df.loc[mask, 'y_coord'] = y + np.random.normal(0, 0.5, mask.sum())
        
        # Fill remaining with US center
        remaining = df['x_coord'].isna()
        if remaining.any():
            df.loc[remaining, 'x_coord'] = -98.5 + np.random.normal(0, 5, remaining.sum())
            df.loc[remaining, 'y_coord'] = 39.8 + np.random.normal(0, 3, remaining.sum())
        
        return df
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and cleanup"""
        # Ensure all required columns exist
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                raise ValueError(f"Required column '{col}' not found after processing")
        
        # Remove any remaining NaN values in critical fields
        critical_fields = ['station_id', 'x_coord', 'y_coord', 'bandwidth_mhz']
        df = df.dropna(subset=critical_fields)
        
        # Ensure proper data types
        df['bandwidth_mhz'] = pd.to_numeric(df['bandwidth_mhz'], errors='coerce')
        df['azimuth_deg'] = pd.to_numeric(df['azimuth_deg'], errors='coerce')
        df['x_coord'] = pd.to_numeric(df['x_coord'], errors='coerce')
        df['y_coord'] = pd.to_numeric(df['y_coord'], errors='coerce')
        
        logger.info(f"Final validation complete: {len(df)} valid stations")
        
        return df
    
    def generate_processing_report(self, df: pd.DataFrame, output_path: str):
        """Generate a report of the processing steps taken"""
        report = {
            'processing_summary': {
                'total_stations': len(df),
                'data_source_format': df['data_source_format'].iloc[0] if 'data_source_format' in df.columns else 'unknown',
                'columns_processed': list(df.columns),
                'bea_regions': df['bea_id'].nunique() if 'bea_id' in df.columns else 0,
                'service_types': df['service_type'].value_counts().to_dict() if 'service_type' in df.columns else {}
            },
            'quality_metrics': {
                'coordinate_completeness': (~df[['x_coord', 'y_coord']].isna().any(axis=1)).mean() * 100,
                'bea_assignment_rate': (~df['bea_id'].isna()).mean() * 100 if 'bea_id' in df.columns else 0,
                'technical_param_completeness': (~df[['bandwidth_mhz', 'azimuth_deg']].isna().any(axis=1)).mean() * 100
            },
            'area_distribution': df['area_type'].value_counts().to_dict() if 'area_type' in df.columns else {},
            'cluster_distribution': {
                'total_clusters': df['cluster'].nunique() if 'cluster' in df.columns else 0,
                'avg_stations_per_cluster': len(df) / df['cluster'].nunique() if 'cluster' in df.columns and df['cluster'].nunique() > 0 else 0
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Processing report saved to: {output_path}")


# Integration function for the enhanced runner
def integrate_with_enhanced_runner(input_file: str, 
                                  bea_geojson: str,
                                  service_type: Optional[str] = None) -> pd.DataFrame:
    """
    Integration point for the enhanced BEA optimization runner
    
    This function can be called from enhanced_bea_optimization_runner.py
    to handle pre-processed station data
    """
    adapter = SmartStationDataAdapter(bea_geojson)
    
    try:
        # Process the data
        processed_df = adapter.process(input_file, service_type)
        
        # Generate processing report
        report_path = Path(input_file).stem + "_processing_report.json"
        adapter.generate_processing_report(processed_df, report_path)
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Failed to process station data: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python smart_station_data_adapter.py <input_csv> [bea_geojson] [service_type]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    bea_geojson = sys.argv[2] if len(sys.argv) > 2 else "bea.geojson"
    service_type = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Process the data
    processed_df = integrate_with_enhanced_runner(input_csv, bea_geojson, service_type)
    
    # Save processed data
    output_file = Path(input_csv).stem + "_processed.csv"
    processed_df.to_csv(output_file, index=False)
    
    print(f"✅ Processed {len(processed_df)} stations")
    print(f"📁 Output saved to: {output_file}")
