# updated_enhanced_bea_optimization_runner.py
"""
Updated Enhanced BEA Optimization Runner
Now with smart detection for pre-processed station data
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, Optional, List
from datetime import datetime
import json
import traceback
import webbrowser

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing modules
from spectrum_optimizer import run_optimizer
import bea_mapper
from bea_professional_visualizer import create_professional_visualizations

# Import data adapters
from fcc_data_adapter import FCCDataAdapter
from bea_market_data_adapter import BEAMarketDataAdapter
from smart_station_data_adapter import SmartStationDataAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UpdatedBEAOptimizationPipeline:
    """
    Updated pipeline that intelligently handles all data formats:
    - Pre-processed station data (like AM_STATION_50_CLEANED.csv)
    - BEA market-level data
    - Raw FCC data
    """
    
    def __init__(self, config_file: str = 'optimization_config.yaml'):
        """
        Initialize the updated pipeline
        """
        self.config = self._load_config(config_file)
        self.fcc_adapter = FCCDataAdapter()
        self.smart_adapter = None  # Initialized when needed
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file or use defaults"""
        default_config = {
            'quality_checks': {
                'required_fields': [
                    'station_id', 'x_coord', 'y_coord', 'bandwidth_mhz',
                    'azimuth_deg', 'cluster', 'state', 'area_type'
                ],
                'coordinate_bounds': {
                    'continental_us': {
                        'x_min': -125, 'x_max': -66,
                        'y_min': 24, 'y_max': 49
                    }
                }
            },
            'optimization': {
                'use_partitioning': True,
                'plot_results': False,
                'solver_timeout': 300
            },
            'visualization': {
                'generate_professional': True,
                'open_in_browser': True
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}. Using defaults.")
        
        return default_config
    
    def detect_data_format(self, input_file: str) -> str:
        """
        Intelligently detect the format of the input data
        
        Returns:
        --------
        str
            'station_processed', 'station_partial', 'bea_market', or 'fcc_raw'
        """
        logger.info("Detecting data format...")
        
        # Load a sample of the data
        df_sample = pd.read_csv(input_file, nrows=100)
        columns = set(df_sample.columns)
        
        # Check for BEA market data
        if 'market' in columns and df_sample['market'].str.contains('BEA\\d{3}', na=False).any():
            return 'bea_market'
        
        # Check for pre-processed station data
        required_station_cols = {'station_id', 'x_coord', 'y_coord', 'bandwidth_mhz', 'azimuth_deg'}
        if required_station_cols.issubset(columns):
            # Check completeness
            if {'cluster', 'state', 'area_type'}.issubset(columns):
                return 'station_processed'
            else:
                return 'station_partial'
        
        # Default to raw FCC data
        return 'fcc_raw'
    
    def process_data(self,
                    input_file: str,
                    service_type: Optional[str] = None,
                    bea_geojson: Optional[str] = None) -> pd.DataFrame:
        """
        Process input data based on detected format
        """
        # Detect format
        data_format = self.detect_data_format(input_file)
        logger.info(f"Detected data format: {data_format}")
        
        # Use default BEA geojson if not provided
        if bea_geojson is None:
            bea_geojson = self._find_bea_geojson()
        
        # Process based on format
        if data_format == 'station_processed' or data_format == 'station_partial':
            # Use smart adapter for pre-processed station data
            logger.info("Using Smart Station Data Adapter")
            if self.smart_adapter is None:
                self.smart_adapter = SmartStationDataAdapter(bea_geojson)
            return self.smart_adapter.process(input_file, service_type)
        
        elif data_format == 'bea_market':
            # Use BEA market adapter
            logger.info("Using BEA Market Data Adapter")
            adapter = BEAMarketDataAdapter(bea_geojson)
            temp_output = f"temp_stations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            try:
                return adapter.process_data(
                    input_file=input_file,
                    output_file=temp_output,
                    station_density='auto'
                )
            finally:
                if os.path.exists(temp_output):
                    os.remove(temp_output)
        
        else:  # fcc_raw
            # Use FCC adapter
            logger.info("Using FCC Data Adapter for raw data")
            return self.fcc_adapter.process_fcc_data(
                input_file=input_file,
                service_type=service_type,
                bea_geojson_path=bea_geojson
            )
    
    def _find_bea_geojson(self) -> str:
        """Find BEA GeoJSON file in common locations"""
        search_paths = [
            "bea.geojson",
            "data/bea.geojson",
            "../data/bea.geojson",
            str(Path(__file__).parent / "bea.geojson"),
            str(Path(__file__).parent.parent / "data" / "bea.geojson")
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                logger.info(f"Found BEA GeoJSON at: {path}")
                return path
        
        logger.warning("BEA GeoJSON not found in common locations")
        return "bea.geojson"  # Default path
    
    def run_optimization_pipeline(self,
                                 input_file: str,
                                 output_dir: str = 'optimization_results',
                                 service_type: Optional[str] = None,
                                 bea_geojson: Optional[str] = None,
                                 **kwargs) -> Dict:
        """
        Run the complete optimization pipeline
        """
        # Create output directory with timestamp
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = output_path / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        logger.info(f"Starting optimization pipeline")
        logger.info(f"Input: {input_file}")
        logger.info(f"Output: {run_dir}")
        
        try:
            # Step 1: Process input data
            logger.info("\n=== Step 1: Processing Input Data ===")
            processed_df = self.process_data(input_file, service_type, bea_geojson)
            
            # Save processed data
            processed_file = run_dir / "processed_data.csv"
            processed_df.to_csv(processed_file, index=False)
            logger.info(f"✅ Processed {len(processed_df)} stations")
            
            # Step 2: Analyze data
            logger.info("\n=== Step 2: Analyzing Data ===")
            analysis = self._analyze_data(processed_df)
            
            # Save analysis
            with open(run_dir / "data_analysis.json", 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Step 3: Run optimization
            logger.info("\n=== Step 3: Running Spectrum Optimization ===")
            optimization_output = run_dir / "optimized_spectrum.csv"
            
            # Use configuration settings
            use_partitioning = kwargs.get('use_partitioning', 
                                        self.config['optimization']['use_partitioning'])
            
            results = run_optimizer(
                input_csv=str(processed_file),
                output_csv=str(optimization_output),
                use_partitioning=use_partitioning,
                plot_results=False  # We'll generate better visualizations
            )
            
            if results is None:
                raise RuntimeError("Optimization failed - check logs above")
            
            logger.info(f"✅ Optimization complete: {len(results)} stations optimized")
            
            # Step 4: Generate visualizations
            if self.config['visualization']['generate_professional']:
                logger.info("\n=== Step 4: Generating Professional Visualizations ===")
                try:
                    viz_output_dir = run_dir / "visualizations"
                    viz_output_dir.mkdir(exist_ok=True)
                    
                    # Change to visualization directory for output
                    original_cwd = os.getcwd()
                    os.chdir(str(viz_output_dir))
                    
                    create_professional_visualizations(
                        str(optimization_output),
                        bea_geojson or self._find_bea_geojson()
                    )
                    
                    os.chdir(original_cwd)
                    
                    # Open in browser if configured
                    if self.config['visualization']['open_in_browser']:
                        report_path = viz_output_dir / "complete_optimization_report.html"
                        if report_path.exists():
                            webbrowser.open(f"file://{report_path.absolute()}")
                            logger.info("📊 Opened visualization report in browser")
                    
                except Exception as e:
                    logger.warning(f"Visualization generation failed: {e}")
                    traceback.print_exc()
            
            # Step 5: Generate summary
            logger.info("\n=== Step 5: Creating Summary ===")
            summary = self._create_summary(processed_df, results, analysis, run_dir)
            
            # Save summary
            with open(run_dir / "optimization_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Print summary to console
            self._print_summary(summary)
            
            logger.info("\n✅ Pipeline complete!")
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            traceback.print_exc()
            
            # Save error report
            error_report = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(run_dir / "error_report.json", 'w') as f:
                json.dump(error_report, f, indent=2)
            
            return {
                'success': False,
                'error': str(e),
                'output_dir': str(run_dir)
            }
    
    def _analyze_data(self, df: pd.DataFrame) -> Dict:
        """Analyze the processed data"""
        analysis = {
            'total_stations': len(df),
            'data_quality': {
                'has_coordinates': (~df[['x_coord', 'y_coord']].isna().any(axis=1)).sum(),
                'has_bea_assignment': (~df['bea_id'].isna()).sum() if 'bea_id' in df.columns else 0,
                'complete_records': (~df.isna().any(axis=1)).sum()
            },
            'geographic_distribution': {
                'states': df['state'].value_counts().to_dict() if 'state' in df.columns else {},
                'bea_regions': df['bea_id'].nunique() if 'bea_id' in df.columns else 0,
                'clusters': df['cluster'].nunique() if 'cluster' in df.columns else 0
            },
            'technical_specs': {
                'bandwidth_stats': {
                    'min': df['bandwidth_mhz'].min(),
                    'max': df['bandwidth_mhz'].max(),
                    'mean': df['bandwidth_mhz'].mean(),
                    'total': df['bandwidth_mhz'].sum()
                } if 'bandwidth_mhz' in df.columns else {},
                'area_types': df['area_type'].value_counts().to_dict() if 'area_type' in df.columns else {}
            }
        }
        
        return analysis
    
    def _create_summary(self, input_df: pd.DataFrame, 
                       results_df: pd.DataFrame,
                       analysis: Dict,
                       run_dir: Path) -> Dict:
        """Create comprehensive summary"""
        # Calculate optimization metrics
        successful = results_df['optimized_start_freq_mhz'].notna().sum()
        total = len(results_df)
        
        summary = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(run_dir),
            'input_analysis': analysis,
            'optimization_results': {
                'total_stations': total,
                'successful_allocations': successful,
                'success_rate': (successful / total * 100) if total > 0 else 0,
                'failed_allocations': total - successful
            }
        }
        
        # Add spectrum efficiency metrics
        if successful > 0:
            freq_min = results_df['optimized_start_freq_mhz'].min()
            freq_max = results_df['optimized_end_freq_mhz'].max()
            spectrum_used = freq_max - freq_min
            bandwidth_total = results_df['bandwidth_mhz'].sum()
            
            summary['spectrum_metrics'] = {
                'frequency_range': [float(freq_min), float(freq_max)],
                'spectrum_used_mhz': float(spectrum_used),
                'bandwidth_allocated_mhz': float(bandwidth_total),
                'spectrum_efficiency_percent': float(bandwidth_total / spectrum_used * 100) if spectrum_used > 0 else 0,
                'frequency_reuse_factor': float(bandwidth_total / spectrum_used) if spectrum_used > 0 else 0
            }
        
        # Add allocation method breakdown
        if 'allocation_method' in results_df.columns:
            summary['allocation_methods'] = results_df['allocation_method'].value_counts().to_dict()
        
        # File locations
        summary['output_files'] = {
            'processed_data': 'processed_data.csv',
            'optimization_results': 'optimized_spectrum.csv',
            'data_analysis': 'data_analysis.json',
            'summary': 'optimization_summary.json'
        }
        
        # Add visualization info if created
        viz_dir = run_dir / "visualizations"
        if viz_dir.exists() and any(viz_dir.iterdir()):
            summary['visualizations'] = {
                'generated': True,
                'main_report': 'visualizations/complete_optimization_report.html',
                'interactive_map': 'visualizations/interactive_station_map.html'
            }
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print summary to console"""
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        if summary.get('success'):
            opt_results = summary.get('optimization_results', {})
            print(f"\n📊 Results:")
            print(f"  • Total stations: {opt_results.get('total_stations', 0):,}")
            print(f"  • Successful allocations: {opt_results.get('successful_allocations', 0):,}")
            print(f"  • Success rate: {opt_results.get('success_rate', 0):.1f}%")
            
            if 'spectrum_metrics' in summary:
                metrics = summary['spectrum_metrics']
                print(f"\n📡 Spectrum Efficiency:")
                print(f"  • Frequency range: {metrics['frequency_range'][0]:.0f}-{metrics['frequency_range'][1]:.0f} MHz")
                print(f"  • Total spectrum used: {metrics['spectrum_used_mhz']:.1f} MHz")
                print(f"  • Total bandwidth allocated: {metrics['bandwidth_allocated_mhz']:.1f} MHz")
                print(f"  • Spectrum efficiency: {metrics['spectrum_efficiency_percent']:.1f}%")
                print(f"  • Frequency reuse factor: {metrics['frequency_reuse_factor']:.1f}x")
            
            print(f"\n📁 Output directory: {summary.get('output_directory', 'N/A')}")
            
            if 'visualizations' in summary and summary['visualizations'].get('generated'):
                print(f"\n🎨 Visualizations generated - check your browser!")
        else:
            print(f"\n❌ Optimization failed: {summary.get('error', 'Unknown error')}")


def main():
    """Main function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Smart BEA Spectrum Optimization Pipeline - Handles all data formats'
    )
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--output-dir', default='optimization_results',
                       help='Output directory (default: optimization_results)')
    parser.add_argument('--service-type', default=None,
                       help='Radio service type (AM, FM, TV, AWS, etc.)')
    parser.add_argument('--bea-geojson', default=None,
                       help='Path to BEA GeoJSON file')
    parser.add_argument('--no-partitioning', action='store_true',
                       help='Disable optimization partitioning')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--config', default='optimization_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = UpdatedBEAOptimizationPipeline(config_file=args.config)
    
    # Override visualization setting if requested
    if args.no_visualizations:
        pipeline.config['visualization']['generate_professional'] = False
    
    # Run pipeline
    print("\n" + "="*60)
    print("SMART BEA SPECTRUM OPTIMIZATION PIPELINE")
    print("="*60)
    print(f"\n📂 Input: {args.input_file}")
    print(f"📁 Output: {args.output_dir}")
    
    results = pipeline.run_optimization_pipeline(
        input_file=args.input_file,
        output_dir=args.output_dir,
        service_type=args.service_type,
        bea_geojson=args.bea_geojson,
        use_partitioning=not args.no_partitioning
    )
    
    return 0 if results.get('success') else 1


if __name__ == "__main__":
    sys.exit(main())
