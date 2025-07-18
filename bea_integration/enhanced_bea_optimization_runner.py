# enhanced_bea_optimization_runner.py
"""
Enhanced BEA Optimization Runner with Real FCC Data Support
Integrates the FCC Data Adapter for seamless processing of real data
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, Optional, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing modules
from core import spectrum_optimizer
from core import Spectrum_Optimizer_Result_Analyzer as analyzer
import bea_mapper
from bea_spectrum_visualizer import BEASpectrumOptimizer
from bea_csv_reader import read_and_analyze_bea_csv

# Import new FCC data adapter
from fcc_data_adapter import FCCDataAdapter, process_fcc_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedBEAOptimizationPipeline:
    """
    Enhanced pipeline that handles real FCC data seamlessly
    """
    
    def __init__(self, config_file: str = 'fcc_data_config.yaml'):
        """
        Initialize the enhanced pipeline
        
        Parameters:
        -----------
        config_file : str
            Path to FCC data configuration file
        """
        self.config = self._load_config(config_file)
        self.fcc_adapter = FCCDataAdapter()
        self.results_cache = {}
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file {config_file} not found, using defaults")
            return {}
    
    def process_fcc_data(self, 
                        input_file: str,
                        data_format: str = 'auto',
                        service_type: Optional[str] = None,
                        bea_geojson: Optional[str] = None) -> pd.DataFrame:
        """
        Process FCC data with format detection and validation
        
        Parameters:
        -----------
        input_file : str
            Path to FCC data file
        data_format : str
            Data format type (am_station, fm_station, aws_license, etc.) or 'auto'
        service_type : str, optional
            Radio service type
        bea_geojson : str, optional
            Path to BEA shapes file
            
        Returns:
        --------
        pd.DataFrame
            Processed data ready for optimization
        """
        logger.info(f"Processing FCC data: {input_file}")
        
        # Load raw data
        raw_df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(raw_df)} records")
        
        # Detect format if auto
        if data_format == 'auto':
            data_format = self._detect_data_format(raw_df)
            logger.info(f"Auto-detected format: {data_format}")
        
        # Apply format-specific mappings
        if data_format in self.config.get('data_formats', {}):
            format_config = self.config['data_formats'][data_format]
            
            # Apply column mappings
            column_mappings = format_config.get('column_mappings', {})
            raw_df = raw_df.rename(columns=column_mappings)
            
            # Apply defaults
            defaults = format_config.get('defaults', {})
            for col, value in defaults.items():
                if col not in raw_df.columns:
                    raw_df[col] = value
            
            # Override service type if specified in format
            if 'service_type' in defaults and service_type is None:
                service_type = defaults['service_type']
        
        # Process with FCC adapter
        processed_df = self.fcc_adapter.process_fcc_data(
            input_file=input_file,
            service_type=service_type,
            bea_geojson_path=bea_geojson
        )
        
        # Validate processed data
        validation_result = self._validate_processed_data(processed_df)
        if not validation_result['valid']:
            logger.error(f"Data validation failed: {validation_result['errors']}")
            raise ValueError("Processed data failed validation")
        
        # Log warnings
        for warning in validation_result['warnings']:
            logger.warning(warning)
        
        return processed_df
    
    def _detect_data_format(self, df: pd.DataFrame) -> str:
        """Auto-detect FCC data format based on columns"""
        columns = set(df.columns)
        
        # Check each known format
        format_scores = {}
        
        for format_name, format_config in self.config.get('data_formats', {}).items():
            mappings = format_config.get('column_mappings', {})
            # Count how many expected columns are present
            matches = sum(1 for col in mappings.keys() if col in columns)
            if matches > 0:
                format_scores[format_name] = matches / len(mappings)
        
        if format_scores:
            # Return format with highest score
            best_format = max(format_scores, key=format_scores.get)
            logger.info(f"Format detection scores: {format_scores}")
            return best_format
        
        return 'generic_fcc'
    
    def _validate_processed_data(self, df: pd.DataFrame) -> Dict:
        """Validate processed data meets requirements"""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = self.config.get('quality_checks', {}).get('required_fields', [])
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
        
        # Check data types
        numeric_fields = ['x_coord', 'y_coord', 'bandwidth_mhz', 'azimuth_deg', 'elevation_deg']
        for field in numeric_fields:
            if field in df.columns:
                non_numeric = df[field].apply(lambda x: not isinstance(x, (int, float, np.number)))
                if non_numeric.any():
                    errors.append(f"{field} contains non-numeric values")
        
        # Check coordinate bounds
        coord_bounds = self.config.get('quality_checks', {}).get('coordinate_bounds', {})
        if 'continental_us' in coord_bounds:
            bounds = coord_bounds['continental_us']
            out_of_bounds = (
                (df['x_coord'] < bounds['x_min']) | (df['x_coord'] > bounds['x_max']) |
                (df['y_coord'] < bounds['y_min']) | (df['y_coord'] > bounds['y_max'])
            )
            
            # Check if they might be in Alaska or Hawaii
            alaska_bounds = coord_bounds.get('alaska', {})
            hawaii_bounds = coord_bounds.get('hawaii', {})
            
            alaska_mask = (
                (df['x_coord'] >= alaska_bounds.get('x_min', -180)) & 
                (df['x_coord'] <= alaska_bounds.get('x_max', -130)) &
                (df['y_coord'] >= alaska_bounds.get('y_min', 52)) & 
                (df['y_coord'] <= alaska_bounds.get('y_max', 72))
            )
            
            hawaii_mask = (
                (df['x_coord'] >= hawaii_bounds.get('x_min', -161)) & 
                (df['x_coord'] <= hawaii_bounds.get('x_max', -154)) &
                (df['y_coord'] >= hawaii_bounds.get('y_min', 18)) & 
                (df['y_coord'] <= hawaii_bounds.get('y_max', 23))
            )
            
            truly_out_of_bounds = out_of_bounds & ~alaska_mask & ~hawaii_mask
            
            if truly_out_of_bounds.any():
                warnings.append(f"{truly_out_of_bounds.sum()} stations have coordinates outside US bounds")
        
        # Check bandwidth limits
        bw_limits = self.config.get('quality_checks', {}).get('bandwidth_limits', {})
        if 'min' in bw_limits and 'max' in bw_limits:
            invalid_bw = (
                (df['bandwidth_mhz'] < bw_limits['min']) | 
                (df['bandwidth_mhz'] > bw_limits['max'])
            )
            if invalid_bw.any():
                warnings.append(f"{invalid_bw.sum()} stations have bandwidth outside normal range")
        
        # Check for duplicates
        duplicates = df['station_id'].duplicated()
        if duplicates.any():
            errors.append(f"{duplicates.sum()} duplicate station IDs found")
        
        # Check for missing critical data
        critical_fields = ['station_id', 'x_coord', 'y_coord', 'bandwidth_mhz']
        for field in critical_fields:
            if field in df.columns:
                missing = df[field].isna()
                if missing.any():
                    errors.append(f"{field} has {missing.sum()} missing values")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def run_optimization_pipeline(self,
                                 input_file: str,
                                 output_dir: str = 'optimization_results',
                                 data_format: str = 'auto',
                                 service_type: Optional[str] = None,
                                 bea_geojson: Optional[str] = None,
                                 **optimization_params) -> Dict:
        """
        Run the complete optimization pipeline
        
        Parameters:
        -----------
        input_file : str
            Path to FCC data file
        output_dir : str
            Directory for output files
        data_format : str
            Data format type or 'auto'
        service_type : str, optional
            Radio service type
        bea_geojson : str, optional
            Path to BEA shapes file
        **optimization_params : dict
            Additional parameters for optimization
            
        Returns:
        --------
        dict
            Results summary
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Add timestamp to outputs
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = output_path / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        logger.info(f"Starting optimization pipeline - Output: {run_dir}")
        
        try:
            # Step 1: Process FCC data
            logger.info("Step 1: Processing FCC data...")
            processed_df = self.process_fcc_data(
                input_file=input_file,
                data_format=data_format,
                service_type=service_type,
                bea_geojson=bea_geojson
            )
            
            # Save processed data
            processed_file = run_dir / "processed_fcc_data.csv"
            processed_df.to_csv(processed_file, index=False)
            logger.info(f"Saved processed data: {processed_file}")
            
            # Step 2: Analyze data characteristics
            logger.info("Step 2: Analyzing data characteristics...")
            analysis_results = self._analyze_data_characteristics(processed_df)
            
            # Save analysis
            with open(run_dir / "data_analysis.txt", 'w') as f:
                f.write("FCC DATA ANALYSIS\n")
                f.write("=" * 60 + "\n\n")
                for key, value in analysis_results.items():
                    f.write(f"{key}: {value}\n")
            
            # Step 3: Initialize optimizer
            logger.info("Step 3: Initializing BEA spectrum optimizer...")
            
            # Use the processed file as input to BEASpectrumOptimizer
            optimizer = BEASpectrumOptimizer(
                bea_geojson_path=bea_geojson or "bea.geojson",
                bea_table_csv_path=str(processed_file)
            )
            
            # Load the data
            optimizer.load_bea_data()
            
            # The data is already in the correct format, so we can skip prepare_optimization_input
            # and use it directly
            optimizer.station_data = processed_df
            
            # Step 4: Run optimization
            logger.info("Step 4: Running spectrum optimization...")
            
            optimization_input = run_dir / "optimization_input.csv"
            optimization_output = run_dir / "optimized_spectrum.csv"
            
            # Save the processed data as optimization input
            processed_df.to_csv(optimization_input, index=False)
            
            # Run optimization with provided parameters
            use_partitioning = optimization_params.get('use_partitioning', True)
            plot_results = optimization_params.get('plot_results', True)
            
            results = spectrum_optimizer.run_optimizer(
                input_csv=str(optimization_input),
                output_csv=str(optimization_output),
                use_partitioning=use_partitioning,
                plot_results=plot_results
            )
            
            if results is None:
                raise RuntimeError("Optimization failed")
            
            # Update optimizer results
            optimizer.optimization_results = results
            
            # Step 5: Generate comprehensive report
            logger.info("Step 5: Generating comprehensive report...")
            report_dir = run_dir / "full_report"
            optimizer.generate_full_report(str(report_dir))
            
            # Step 6: Create summary
            summary = self._create_optimization_summary(
                processed_df, results, analysis_results, run_dir
            )
            
            # Save summary
            with open(run_dir / "optimization_summary.json", 'w') as f:
                import json
                json.dump(summary, f, indent=2)
            
            logger.info("✅ Optimization pipeline complete!")
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Save error report
            with open(run_dir / "error_report.txt", 'w') as f:
                f.write(f"Error: {str(e)}\n\n")
                f.write("Traceback:\n")
                traceback.print_exc(file=f)
            
            return {
                'success': False,
                'error': str(e),
                'output_dir': str(run_dir)
            }
    
    def _analyze_data_characteristics(self, df: pd.DataFrame) -> Dict:
        """Analyze characteristics of the processed data"""
        analysis = {}
        
        # Basic statistics
        analysis['total_stations'] = len(df)
        analysis['unique_licensees'] = df['licensee'].nunique() if 'licensee' in df.columns else 0
        analysis['service_types'] = df['service_type'].value_counts().to_dict() if 'service_type' in df.columns else {}
        
        # Geographic distribution
        analysis['states'] = df['state'].value_counts().to_dict() if 'state' in df.columns else {}
        analysis['bea_regions'] = df['bea_id'].nunique() if 'bea_id' in df.columns else 0
        analysis['clusters'] = df['cluster'].nunique() if 'cluster' in df.columns else 0
        
        # Area type distribution
        if 'area_type' in df.columns:
            area_dist = df['area_type'].value_counts()
            analysis['area_distribution'] = area_dist.to_dict()
            analysis['urban_percentage'] = (area_dist.get('urban', 0) / len(df) * 100)
        
        # Bandwidth analysis
        if 'bandwidth_mhz' in df.columns:
            analysis['bandwidth_stats'] = {
                'mean': df['bandwidth_mhz'].mean(),
                'min': df['bandwidth_mhz'].min(),
                'max': df['bandwidth_mhz'].max(),
                'total': df['bandwidth_mhz'].sum()
            }
        
        # Data quality
        coord_estimated = df.get('coord_estimated', pd.Series([False] * len(df)))
        analysis['estimated_coordinates'] = coord_estimated.sum()
        analysis['data_quality_score'] = df.get('data_quality_score', pd.Series([1.0] * len(df))).mean()
        
        return analysis
    
    def _create_optimization_summary(self, 
                                   input_df: pd.DataFrame,
                                   results_df: pd.DataFrame,
                                   analysis: Dict,
                                   run_dir: Path) -> Dict:
        """Create comprehensive summary of optimization results"""
        summary = {
            'success': True,
            'timestamp': pd.Timestamp.now().isoformat(),
            'output_dir': str(run_dir),
            
            'input_data': {
                'total_stations': len(input_df),
                'data_characteristics': analysis
            },
            
            'optimization_results': {
                'stations_optimized': len(results_df),
                'successful_allocations': results_df['optimized_start_freq_mhz'].notna().sum(),
                'failed_allocations': results_df['optimized_start_freq_mhz'].isna().sum()
            }
        }
        
        # Spectrum efficiency
        if results_df['optimized_start_freq_mhz'].notna().any():
            min_freq = results_df['optimized_start_freq_mhz'].min()
            max_freq = results_df['optimized_end_freq_mhz'].max()
            total_spectrum = max_freq - min_freq
            total_bandwidth = results_df['bandwidth_mhz'].sum()
            
            summary['spectrum_metrics'] = {
                'frequency_range': [min_freq, max_freq],
                'total_spectrum_used': total_spectrum,
                'total_bandwidth_allocated': total_bandwidth,
                'spectrum_efficiency': (total_bandwidth / total_spectrum * 100) if total_spectrum > 0 else 0
            }
        
        # Allocation methods
        if 'allocation_method' in results_df.columns:
            method_counts = results_df['allocation_method'].value_counts()
            summary['allocation_methods'] = method_counts.to_dict()
        
        # Output files
        summary['output_files'] = {
            'processed_data': 'processed_fcc_data.csv',
            'optimization_input': 'optimization_input.csv',
            'optimization_results': 'optimized_spectrum.csv',
            'report_directory': 'full_report/',
            'interactive_map': 'full_report/interactive_map.html'
        }
        
        return summary


def main():
    """Main function to demonstrate the enhanced pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced BEA Spectrum Optimization Pipeline')
    parser.add_argument('input_file', help='Path to FCC data file')
    parser.add_argument('--output-dir', default='optimization_results', 
                       help='Output directory (default: optimization_results)')
    parser.add_argument('--data-format', default='auto',
                       help='Data format type (auto, am_station, fm_station, aws_license, etc.)')
    parser.add_argument('--service-type', default=None,
                       help='Radio service type (AM, FM, TV, AWS, etc.)')
    parser.add_argument('--bea-geojson', default=None,
                       help='Path to BEA GeoJSON file')
    parser.add_argument('--config-file', default='fcc_data_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-partitioning', action='store_true',
                       help='Disable optimization partitioning')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EnhancedBEAOptimizationPipeline(config_file=args.config_file)
    
    # Run optimization
    print("=" * 80)
    print("ENHANCED BEA SPECTRUM OPTIMIZATION PIPELINE")
    print("=" * 80)
    print(f"\nInput file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data format: {args.data_format}")
    print(f"Service type: {args.service_type or 'Auto-detect'}")
    print()
    
    # Run the pipeline
    results = pipeline.run_optimization_pipeline(
        input_file=args.input_file,
        output_dir=args.output_dir,
        data_format=args.data_format,
        service_type=args.service_type,
        bea_geojson=args.bea_geojson,
        use_partitioning=not args.no_partitioning,
        plot_results=not args.no_plots
    )
    
    # Print results
    if results.get('success'):
        print("\n✅ OPTIMIZATION COMPLETE!")
        print(f"\nResults saved to: {results['output_dir']}")
        
        if 'optimization_results' in results:
            opt_results = results['optimization_results']
            print(f"\nOptimization Summary:")
            print(f"  - Stations processed: {opt_results['stations_optimized']}")
            print(f"  - Successful allocations: {opt_results['successful_allocations']}")
            print(f"  - Failed allocations: {opt_results['failed_allocations']}")
        
        if 'spectrum_metrics' in results:
            metrics = results['spectrum_metrics']
            print(f"\nSpectrum Metrics:")
            print(f"  - Frequency range: {metrics['frequency_range'][0]:.1f} - {metrics['frequency_range'][1]:.1f} MHz")
            print(f"  - Total spectrum used: {metrics['total_spectrum_used']:.1f} MHz")
            print(f"  - Spectrum efficiency: {metrics['spectrum_efficiency']:.1f}%")
        
        print(f"\nKey outputs:")
        if 'output_files' in results:
            files = results['output_files']
            print(f"  - Processed data: {results['output_dir']}/{files['processed_data']}")
            print(f"  - Optimization results: {results['output_dir']}/{files['optimization_results']}")
            print(f"  - Interactive map: {results['output_dir']}/{files['interactive_map']}")
            print(f"  - Full report: {results['output_dir']}/{files['report_directory']}")
    else:
        print(f"\n❌ OPTIMIZATION FAILED!")
        print(f"Error: {results.get('error', 'Unknown error')}")
        print(f"Check error report at: {results['output_dir']}/error_report.txt")
    
    return 0 if results.get('success') else 1


if __name__ == "__main__":
    sys.exit(main())
