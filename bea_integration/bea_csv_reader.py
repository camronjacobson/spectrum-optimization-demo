"""
BEA CSV File Reader and Analyzer
Reads and analyzes the BEA table CSV file to understand station data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def read_and_analyze_bea_csv(csv_path):
    """
    Read and analyze the BEA CSV file
    
    Parameters:
    -----------
    csv_path : str
        Path to the BEA CSV file
    
    Returns:
    --------
    pd.DataFrame
        The loaded dataframe with analysis
    """
    print("=" * 80)
    print("BEA CSV FILE ANALYSIS")
    print("=" * 80)
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Basic information
    print(f"\n📊 BASIC INFORMATION:")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    
    # Check for missing values
    print(f"\n🔍 MISSING VALUES:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"  - {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Analyze each column
    print(f"\n📈 COLUMN ANALYSIS:")
    
    # Callsigns
    print(f"\n1. Callsigns:")
    print(f"   - Unique callsigns: {df['callsigns'].nunique()}")
    print(f"   - Sample callsigns: {df['callsigns'].head(5).tolist()}")
    
    # Radio Service
    if 'radio service' in df.columns:
        print(f"\n2. Radio Service Distribution:")
        service_counts = df['radio service'].value_counts()
        for service, count in service_counts.head(10).items():
            print(f"   - {service}: {count} ({count/len(df)*100:.1f}%)")
    
    # Status
    if 'status' in df.columns:
        print(f"\n3. License Status:")
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items():
            print(f"   - {status}: {count} ({count/len(df)*100:.1f}%)")
    
    # Expiration dates
    if 'expiration date' in df.columns:
        print(f"\n4. Expiration Dates:")
        # Convert to datetime
        df['expiration_datetime'] = pd.to_datetime(df['expiration date'], errors='coerce')
        valid_dates = df['expiration_datetime'].notna()
        
        if valid_dates.any():
            print(f"   - Earliest expiration: {df.loc[valid_dates, 'expiration_datetime'].min()}")
            print(f"   - Latest expiration: {df.loc[valid_dates, 'expiration_datetime'].max()}")
            print(f"   - Invalid date entries: {(~valid_dates).sum()}")
            
            # Group by year
            df['expiration_year'] = df['expiration_datetime'].dt.year
            year_counts = df['expiration_year'].value_counts().sort_index()
            print(f"\n   Expirations by year:")
            for year, count in year_counts.head(5).items():
                if pd.notna(year):
                    print(f"   - {int(year)}: {count}")
    
    # Name analysis
    if 'name' in df.columns:
        print(f"\n5. Licensee Names:")
        print(f"   - Unique licensees: {df['name'].nunique()}")
        # Top licensees
        top_licensees = df['name'].value_counts().head(5)
        print(f"   - Top 5 licensees:")
        for name, count in top_licensees.items():
            print(f"     • {name}: {count} licenses")
    
    # Extract location information from callsigns (if pattern exists)
    print(f"\n6. Geographic Distribution (from callsigns):")
    # Common callsign prefixes might indicate location
    prefixes = df['callsigns'].str[:1].value_counts()
    print(f"   - Callsign prefix distribution:")
    for prefix, count in prefixes.head(5).items():
        print(f"     • {prefix}: {count} stations")
    
    return df


def visualize_bea_data(df, output_dir="."):
    """
    Create visualizations of the BEA data
    
    Parameters:
    -----------
    df : pd.DataFrame
        The BEA dataframe
    output_dir : str
        Directory to save visualizations
    """
    print(f"\n📊 CREATING VISUALIZATIONS...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Radio Service Distribution
    if 'radio service' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        service_counts = df['radio service'].value_counts().head(15)
        service_counts.plot(kind='barh', ax=ax)
        ax.set_xlabel('Number of Licenses')
        ax.set_title('Top 15 Radio Services by License Count', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/radio_service_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Status Distribution
    if 'status' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 8))
        status_counts = df['status'].value_counts()
        colors = plt.cm.Set3(range(len(status_counts)))
        status_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors)
        ax.set_ylabel('')
        ax.set_title('License Status Distribution', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/license_status_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Expiration Timeline
    if 'expiration_datetime' in df.columns:
        valid_dates = df[df['expiration_datetime'].notna()].copy()
        if len(valid_dates) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Group by year and month
            valid_dates['year_month'] = valid_dates['expiration_datetime'].dt.to_period('M')
            timeline = valid_dates['year_month'].value_counts().sort_index()
            
            # Convert to timestamp for plotting
            timeline.index = timeline.index.to_timestamp()
            timeline.plot(kind='line', ax=ax, linewidth=2)
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Licenses Expiring')
            ax.set_title('License Expiration Timeline', fontsize=14, weight='bold')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/expiration_timeline.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Top Licensees
    if 'name' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        top_licensees = df['name'].value_counts().head(20)
        top_licensees.plot(kind='barh', ax=ax)
        ax.set_xlabel('Number of Licenses')
        ax.set_title('Top 20 Licensees by Number of Licenses', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_licensees.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Visualizations saved to {output_dir}/")


def extract_geographic_hints(df):
    """
    Try to extract geographic information from the data
    
    Parameters:
    -----------
    df : pd.DataFrame
        The BEA dataframe
    
    Returns:
    --------
    dict
        Geographic analysis results
    """
    geo_hints = {}
    
    # Analyze callsigns for state codes
    # Many callsigns contain state abbreviations
    state_codes = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                   'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                   'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                   'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                   'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
    
    state_counts = Counter()
    
    for callsign in df['callsigns']:
        if pd.notna(callsign):
            # Check if state code appears in callsign
            for state in state_codes:
                if state in callsign.upper():
                    state_counts[state] += 1
    
    geo_hints['state_mentions'] = dict(state_counts.most_common(10))
    
    # Analyze names for city/state information
    if 'name' in df.columns:
        city_mentions = Counter()
        major_cities = ['NEW YORK', 'LOS ANGELES', 'CHICAGO', 'HOUSTON', 'PHOENIX',
                       'PHILADELPHIA', 'SAN ANTONIO', 'SAN DIEGO', 'DALLAS', 'SAN JOSE',
                       'AUSTIN', 'JACKSONVILLE', 'FORT WORTH', 'COLUMBUS', 'CHARLOTTE']
        
        for name in df['name']:
            if pd.notna(name):
                name_upper = name.upper()
                for city in major_cities:
                    if city in name_upper:
                        city_mentions[city] += 1
        
        geo_hints['city_mentions'] = dict(city_mentions.most_common(10))
    
    return geo_hints


def prepare_for_bea_mapping(df, bea_shapes_path=None):
    """
    Prepare the data for BEA region mapping
    
    Parameters:
    -----------
    df : pd.DataFrame
        The BEA dataframe
    bea_shapes_path : str
        Path to BEA shapes file (optional)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame ready for BEA mapping
    """
    print(f"\n🗺️  PREPARING DATA FOR BEA MAPPING...")
    
    # Create a mapping dataframe
    mapping_df = pd.DataFrame()
    
    # Use callsign as unique identifier
    mapping_df['station_id'] = df['callsigns']
    
    # Extract service type
    if 'radio service' in df.columns:
        mapping_df['service_type'] = df['radio service']
    
    # License holder
    if 'name' in df.columns:
        mapping_df['licensee'] = df['name']
    
    # Status
    if 'status' in df.columns:
        mapping_df['license_status'] = df['status']
    
    # Expiration
    if 'expiration date' in df.columns:
        mapping_df['expiration_date'] = df['expiration date']
    
    # Add placeholder coordinates (to be filled with actual data)
    # In reality, you would get these from FCC ULS database or other sources
    mapping_df['needs_coordinates'] = True
    
    print(f"✅ Prepared {len(mapping_df)} stations for BEA mapping")
    print(f"⚠️  Note: Actual coordinates needed for geographic mapping")
    
    return mapping_df


# Example usage
if __name__ == "__main__":
    # Read and analyze the BEA CSV
    csv_path = "example_bea_table.csv"
    
    # Load and analyze
    df = read_and_analyze_bea_csv(csv_path)
    
    # Create visualizations
    visualize_bea_data(df)
    
    # Extract geographic hints
    geo_hints = extract_geographic_hints(df)
    print(f"\n🌍 GEOGRAPHIC HINTS:")
    print(f"State mentions in callsigns: {geo_hints['state_mentions']}")
    if 'city_mentions' in geo_hints:
        print(f"City mentions in names: {geo_hints['city_mentions']}")
    
    # Prepare for BEA mapping
    mapping_df = prepare_for_bea_mapping(df)
    
    # Save prepared data
    mapping_df.to_csv("stations_for_bea_mapping.csv", index=False)
    print(f"\n✅ Analysis complete! Check generated files for results.")
