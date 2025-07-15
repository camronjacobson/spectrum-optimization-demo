# example_fcc_optimization.py
"""
Example script showing how to use the enhanced BEA optimization system
with real FCC data
"""

import os
import sys
from pathlib import Path

# Add the project directories to Python path
project_root = Path(__file__).parent.parent  # Go up one level from bea_integration
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'bea_integration'))
sys.path.append(str(project_root / 'core'))

from enhanced_bea_optimization_runner import EnhancedBEAOptimizationPipeline
from fcc_data_adapter import FCCDataAdapter


def example_1_basic_processing():
    """Example 1: Basic FCC data processing"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic FCC Data Processing")
    print("="*60)
    
    # Initialize the FCC data adapter
    adapter = FCCDataAdapter()
    
    # Process a sample FCC file
    # This assumes you have a file called 'sample_fcc_data.csv'
    input_file = "sample_fcc_data.csv"
    
    if not os.path.exists(input_file):
        print(f"Creating sample FCC data file: {input_file}")
        create_sample_fcc_data(input_file)
    
    # Process the data
    processed_df = adapter.process_fcc_data(
        input_file=input_file,
        service_type="AWS",  # or None for auto-detection
        bea_geojson_path="../data/bea.geojson"  # Go up one directory
    )
    
    # Save processed data
    output_file = "processed_fcc_data.csv"
    processed_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Processed {len(processed_df)} stations")
    print(f"📁 Saved to: {output_file}")
    
    # Show summary
    print("\nData Summary:")
    print(f"  - BEA regions: {processed_df['bea_id'].nunique()}")
    print(f"  - Total bandwidth: {processed_df['bandwidth_mhz'].sum():.1f} MHz")
    print(f"  - Area types: {processed_df['area_type'].value_counts().to_dict()}")


def example_2_full_pipeline():
    """Example 2: Full optimization pipeline"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Full Optimization Pipeline")
    print("="*60)
    
    # Initialize the pipeline
    pipeline = EnhancedBEAOptimizationPipeline(config_file="../fcc_data_config.yaml")
    
    # Run the complete pipeline
    results = pipeline.run_optimization_pipeline(
        input_file="sample_fcc_data.csv",
        output_dir="example_optimization_results",
        data_format="auto",  # Auto-detect format
        service_type=None,   # Auto-detect service
        bea_geojson="../data/bea.geojson",
        use_partitioning=True,
        plot_results=True
    )
    
    if results['success']:
        print("\n✅ Optimization successful!")
        print(f"📁 Results in: {results['output_dir']}")
    else:
        print(f"\n❌ Optimization failed: {results['error']}")


def example_3_specific_service():
    """Example 3: Process specific radio service data"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Service-Specific Processing (FM Radio)")
    print("="*60)
    
    # Create sample FM station data
    fm_file = "sample_fm_stations.csv"
    create_sample_fm_data(fm_file)
    
    # Process FM-specific data
    adapter = FCCDataAdapter()
    processed_df = adapter.process_fcc_data(
        input_file=fm_file,
        service_type="FM"
    )
    
    print(f"\n✅ Processed {len(processed_df)} FM stations")
    print("\nFM Station Summary:")
    print(f"  - Frequency range: 88-108 MHz")
    print(f"  - Channel bandwidth: {processed_df['bandwidth_mhz'].iloc[0]} MHz")
    print(f"  - Coverage area types: {processed_df['area_type'].value_counts().to_dict()}")


def example_4_batch_processing():
    """Example 4: Batch process multiple files"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Processing Multiple Files")
    print("="*60)
    
    # List of files to process
    files = [
        ("am_stations.csv", "AM"),
        ("fm_stations.csv", "FM"),
        ("aws_licenses.csv", "AWS")
    ]
    
    adapter = FCCDataAdapter()
    all_results = []
    
    for filename, service_type in files:
        if not os.path.exists("../data/bea.geojson"):
            print(f"Skipping {filename} (not found)")
            continue
            
        print(f"\nProcessing {filename}...")
        try:
            processed_df = adapter.process_fcc_data(
                input_file=filename,
                service_type=service_type
            )
            
            all_results.append({
                'file': filename,
                'service': service_type,
                'stations': len(processed_df),
                'bandwidth': processed_df['bandwidth_mhz'].sum()
            })
            
            # Save individual results
            output_name = f"processed_{filename}"
            processed_df.to_csv(output_name, index=False)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    print("\n📊 Batch Processing Summary:")
    for result in all_results:
        print(f"  - {result['service']}: {result['stations']} stations, "
              f"{result['bandwidth']:.1f} MHz total")


def create_sample_fcc_data(filename):
    """Create a sample FCC data file for testing"""
    import pandas as pd
    import numpy as np
    
    # Create sample data that mimics FCC format
    n_stations = 100
    
    data = {
        'callsigns': [f'K{i:04d}' for i in range(n_stations)],
        'name': [f'Licensee {i//10}' for i in range(n_stations)],
        'radio service': np.random.choice(['AWS-4', 'AWS-3', 'AWS-1'], n_stations),
        'status': 'Active',
        'expiration date': '2030-12-31',
        'FRN': [f'00{i:08d}' for i in range(n_stations)]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created sample file: {filename}")


def create_sample_fm_data(filename):
    """Create sample FM station data"""
    import pandas as pd
    import numpy as np
    
    n_stations = 50
    
    # FM stations typically have city/state in licensee name
    cities = ['New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 'Phoenix, AZ']
    
    # Fixed the f-string formatting issue here
    data = {
        'call_sign': [],
        'licensee': [f'FM Broadcasting Inc., {cities[i % len(cities)]}' for i in range(n_stations)],
        'frequency': np.random.uniform(88.1, 107.9, n_stations),
        'city': [cities[i % len(cities)].split(',')[0] for i in range(n_stations)],
        'state': [cities[i % len(cities)].split(',')[1].strip() for i in range(n_stations)],
        'erp_kw': np.random.uniform(0.1, 100, n_stations)
    }
    
    # Generate call signs separately to avoid the f-string issue
    for i in range(n_stations):
        prefix = 'K' if i > 25 else 'W'
        letter1 = chr(65 + i % 26)
        letter2 = chr(65 + (i//26) % 26)
        call_sign = f'{prefix}{letter1}{letter2}FM'
        data['call_sign'].append(call_sign)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created sample FM file: {filename}")


def main():
    """Run all examples"""
    print("FCC DATA PROCESSING EXAMPLES")
    print("============================")
    
    # Check for required files
    if not os.path.exists("../bea.geojson"):
        print("\n⚠️  Warning: bea.geojson not found in parent directory")
        print("The BEA shapes file is needed for geographic assignment.")
        print("Continuing with default region assignment...\n")
    
    # Run examples
    try:
        example_1_basic_processing()
    except Exception as e:
        print(f"Example 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_2_full_pipeline()
    except Exception as e:
        print(f"Example 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_3_specific_service()
    except Exception as e:
        print(f"Example 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETE")
    print("="*60)
    print("\nTo process your own FCC data:")
    print("1. Place your FCC data file in CSV format")
    print("2. Run: python enhanced_bea_optimization_runner.py your_data.csv")
    print("3. Check the results in the output directory")
    print("\nFor more options, run: python enhanced_bea_optimization_runner.py --help")


if __name__ == "__main__":
    main()
