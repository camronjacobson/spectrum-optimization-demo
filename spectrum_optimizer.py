import pandas as pd
import numpy as np
import networkx as nx
from ortools.sat.python import cp_model
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import math
import time
from datetime import datetime
import json
import logging

# ---------------------------
# CONFIGURATION PARAMETERS
# ---------------------------

# Default channel step in MHz
CHANNEL_STEP = 5

# Global spectrum pool range in MHz
GLOBAL_LOW_FREQ = 2400
GLOBAL_HIGH_FREQ = 2500

# Thresholds for conflicts - now configurable by area type
AREA_DISTANCE_THRESHOLDS = {
    "urban": 5,
    "suburban": 30,
    "rural": 50
}

# Antenna pattern thresholds
AZIMUTH_THRESHOLD_DEG = 90
ELEVATION_THRESHOLD_DEG = 30  # New: elevation angle consideration

# Solver timeout (seconds)
SOLVER_TIMEOUT = 300

# Performance thresholds for large datasets
MAX_STATIONS_PER_COMPONENT = 500  # Use heuristic if larger
SPATIAL_INDEX_THRESHOLD = 1000  # Use spatial indexing above this

# All US states for nationwide support
US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

# Example LICENSE_BANDS - now includes all US states with reasonable allocations
# In practice, these would come from actual FCC license data
LICENSE_BANDS = {
    state: {"low_freq": 2400 + (i % 10) * 10, "high_freq": 2410 + (i % 10) * 10}
    for i, state in enumerate(US_STATES)
}

# ---------------------------
# LOGGING SETUP
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance (in km) between two points on the Earth
    given lat/lon in decimal degrees using the Haversine formula.
    More accurate than the simplified method for large distances.
    """
    R = 6371  # Earth radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat / 2) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def load_license_bands(filepath=None):
    """
    Load license band configuration from file or use defaults.
    Allows for easy extension to real FCC data.
    """
    if filepath:
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load license bands from {filepath}: {e}")
            logger.info("Using default license bands")
    
    return LICENSE_BANDS


# --------------------------
# TOOL SUMMARY
# --------------------------

def output_tool_summary():
    """
    Outputs a high-level summary of the spectrum optimization tool capabilities.
    Enhanced with additional details about improvements.
    """
    print("=" * 80)
    print("SPECTRUM OPTIMIZATION TOOL - HIGH LEVEL SUMMARY")
    print("=" * 80)
    print()
    print("PURPOSE:")
    print("  This tool performs advanced spectrum optimization for radio services to")
    print("  minimize total spectrum usage while ensuring interference-free operation.")
    print()
    print("KEY CAPABILITIES:")
    print("  • Constraint-based optimization using geographic distance and antenna patterns")
    print("  • Global spectrum reuse across multiple license areas and clusters")
    print("  • Dynamic band allocation beyond fixed state-based assignments")
    print("  • Interference prevention through spatial and directional analysis")
    print("  • Multi-level optimization: local (cluster) + global (cross-cluster)")
    print("  • Scalable partitioning for nationwide datasets")
    print("  • Support for elevation angles and 3D antenna patterns")
    print()
    print("OPTIMIZATION APPROACH:")
    print("  1. Conflict Analysis: Build interference graph based on distance/antenna data")
    print("  2. Partitioning: Separate into independent conflict zones")
    print("  3. Local Optimization: Optimize frequency allocation within each cluster")
    print("  4. Global Optimization: Enable spectrum reuse across non-interfering clusters")
    print("  5. Cross-band Coordination: Dynamically allocate from global spectrum pool")
    print()
    print("CONSTRAINTS ENFORCED:")
    print(f"  • Distance thresholds by area type: {AREA_DISTANCE_THRESHOLDS}")
    print(f"  • Maximum azimuth difference: {AZIMUTH_THRESHOLD_DEG} degrees")
    print(f"  • Maximum elevation difference: {ELEVATION_THRESHOLD_DEG} degrees")
    print(f"  • Channel step size: {CHANNEL_STEP} MHz")
    print("  • Bandwidth requirements per station")
    print("  • License band boundaries (when applicable)")
    print("  • Fixed frequency assignments (if specified)")
    print()
    print("INPUTS PROCESSED:")
    print("  • Station coordinates (lat/lon for accurate distance calculations)")
    print("  • Bandwidth requirements per station")
    print("  • Antenna azimuth and elevation data")
    print("  • Cluster/license area assignments")
    print("  • Area type (urban/suburban/rural)")
    print("  • Optional: Fixed frequency assignments")
    print()
    print("OPTIMIZATION GOAL:")
    print("  Minimize total spectrum usage by maximizing frequency reuse across")
    print("  as many licenses and stations as possible while preventing interference.")
    print()
    print("PERFORMANCE FEATURES:")
    print("  • Spatial indexing for large datasets")
    print("  • Heuristic fallback for very large conflict groups")
    print("  • Configurable solver timeouts")
    print("  • Detailed performance metrics and analysis")
    print()
    print("=" * 80)
    print()

# --------------------------
# DATA LOAD
# --------------------------

def load_data(filepath):
    """
    Load and validate data from CSV file.
    Enhanced with better error handling and validation.
    """
    start_time = time.time()
    logger.info(f"Loading data from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # Validate required columns
    required_columns = ['station_id', 'x_coord', 'y_coord', 'bandwidth_mhz',
                       'azimuth_deg', 'cluster']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Add state column if not present
    if 'state' not in df.columns:
        df['state'] = df['cluster'].apply(lambda x: x.split('_')[0])
    
    # Add elevation if not present (default to 0)
    if 'elevation_deg' not in df.columns:
        logger.info("No elevation data found, defaulting to 0 degrees")
        df['elevation_deg'] = 0
    
    # Add area_type if not present
    if 'area_type' not in df.columns:
        # Example mapping - in practice this would come from data or external source
        cluster_area_map = {}
        for cluster in df['cluster'].unique():
            if '_0' in cluster:
                cluster_area_map[cluster] = "urban"
            elif '_1' in cluster:
                cluster_area_map[cluster] = "suburban"
            else:
                cluster_area_map[cluster] = "rural"
        
        df["area_type"] = df["cluster"].map(cluster_area_map).fillna("rural")
    
    # Add fixed_frequency columns if not present (optional feature)
    if 'fixed_start_freq' not in df.columns:
        df['fixed_start_freq'] = np.nan
    if 'fixed_end_freq' not in df.columns:
        df['fixed_end_freq'] = np.nan
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Data loaded in {elapsed:.2f} seconds. {len(df)} stations found.")
    logger.info(f"States represented: {sorted(df['state'].unique())}")
    logger.info(f"Area types: {df['area_type'].value_counts().to_dict()}")
    
    return df


# --------------------------
# IMPROVED CONFLICT DETECTION
# --------------------------

def check_antenna_interference(row1, row2, azimuth_threshold=AZIMUTH_THRESHOLD_DEG,
                             elevation_threshold=ELEVATION_THRESHOLD_DEG):
    """
    Check if two stations' antenna patterns overlap sufficiently to cause interference.
    Now includes elevation angle consideration.
    """
    # Azimuth difference (circular)
    az_diff = min(
        abs(row1["azimuth_deg"] - row2["azimuth_deg"]),
        360 - abs(row1["azimuth_deg"] - row2["azimuth_deg"])
    )
    
    # Elevation difference (if available)
    elev_diff = abs(row1.get("elevation_deg", 0) - row2.get("elevation_deg", 0))
    
    # Both azimuth and elevation must be within thresholds for interference
    return az_diff < azimuth_threshold and elev_diff < elevation_threshold


def build_global_conflict_matrix(df, use_spatial_index=None):
    """
    Builds a global conflict matrix for all stations across all clusters.
    Enhanced with spatial indexing for large datasets and proper distance calculation.
    """
    start_time = time.time()
    n_stations = len(df)
    
    # Auto-determine whether to use spatial indexing
    if use_spatial_index is None:
        use_spatial_index = n_stations > SPATIAL_INDEX_THRESHOLD
    
    logger.info(f"🔍 Building global conflict matrix for {n_stations} stations...")
    logger.info(f"Using spatial indexing: {use_spatial_index}")
    
    conflicts = {}
    indices = df.index.tolist()
    valid_indices = set(indices)  # For quick lookup
    
    if use_spatial_index:
        # Use KD-tree for efficient neighbor search
        coords = np.radians(df[["y_coord", "x_coord"]].values)
        tree = cKDTree(coords)
        
        # Convert max threshold distance to radians (approximate)
        max_threshold_km = max(AREA_DISTANCE_THRESHOLDS.values())
        angular_threshold = max_threshold_km / 6371  # Earth radius in km
        
        checked_pairs = set()
        
        for i, idx_i in enumerate(indices):
            # Find potential neighbors within max threshold
            neighbors_idx = tree.query_ball_point(coords[i], angular_threshold)
            
            for j_idx in neighbors_idx:
                if j_idx <= i:
                    continue
                
                if j_idx >= len(indices):  # Bounds check
                    continue
                    
                idx_j = indices[j_idx]
                
                # Ensure both indices are valid
                if idx_i not in valid_indices or idx_j not in valid_indices:
                    continue
                
                pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                
                row1 = df.loc[idx_i]
                row2 = df.loc[idx_j]
                
                # Use accurate haversine distance
                distance_km = haversine_distance(
                    row1["y_coord"], row1["x_coord"],
                    row2["y_coord"], row2["x_coord"]
                )
                
                # Get appropriate threshold for this pair
                threshold_i = AREA_DISTANCE_THRESHOLDS.get(row1["area_type"], 50)
                threshold_j = AREA_DISTANCE_THRESHOLDS.get(row2["area_type"], 50)
                pair_threshold = min(threshold_i, threshold_j)
                
                # Check both distance and antenna pattern
                if distance_km < pair_threshold:
                    if check_antenna_interference(row1, row2):
                        conflicts[(idx_i, idx_j)] = True
    
    else:
        # Fallback to pairwise comparison for smaller datasets
        for i, idx_i in enumerate(indices):
            for j in range(i + 1, len(indices)):
                idx_j = indices[j]
                
                # Ensure both indices are valid
                if idx_i not in valid_indices or idx_j not in valid_indices:
                    continue
                    
                row1 = df.loc[idx_i]
                row2 = df.loc[idx_j]
                
                distance_km = haversine_distance(
                    row1["y_coord"], row1["x_coord"],
                    row2["y_coord"], row2["x_coord"]
                )
                
                threshold_i = AREA_DISTANCE_THRESHOLDS.get(row1["area_type"], 50)
                threshold_j = AREA_DISTANCE_THRESHOLDS.get(row2["area_type"], 50)
                pair_threshold = min(threshold_i, threshold_j)
                
                if distance_km < pair_threshold:
                    if check_antenna_interference(row1, row2):
                        conflicts[(idx_i, idx_j)] = True
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Found {len(conflicts)} conflicting station pairs in {elapsed:.2f} seconds")
    
    return conflicts


# --------------------------
# PARTITIONING WITH PERFORMANCE OPTIMIZATION
# --------------------------

def partition_and_optimize(df, license_bands=None):
    """
    Partitions dataset into independent connected components and optimizes each.
    Enhanced with better handling of large components.
    """
    logger.info("\n🔗 Partitioning global conflict graph...")
    
    # Load license bands if not provided
    if license_bands is None:
        license_bands = load_license_bands()
    
    # Build global conflict matrix
    global_conflicts = build_global_conflict_matrix(df)
    
    # Build graph from conflicts
    G = nx.Graph()
    G.add_nodes_from(df.index)
    
    for (i, j) in global_conflicts.keys():
        G.add_edge(i, j)
    
    # Find connected components
    components = list(nx.connected_components(G))
    logger.info(f"✅ Found {len(components)} independent conflict zones")
    
    # Sort components by size for better progress tracking
    components = sorted(components, key=len, reverse=True)
    
    optimized_chunks = []
    failed_components = []
    
    for i, component in enumerate(components):
        component_size = len(component)
        logger.info(f"\n🔍 Optimizing conflict group {i+1}/{len(components)} "
                   f"with {component_size} stations")
        
        df_component = df.loc[list(component)].copy()
        
        # Check if we need special handling for large components
        if component_size > MAX_STATIONS_PER_COMPONENT:
            logger.warning(f"Component size ({component_size}) exceeds threshold. "
                          f"Using hierarchical optimization.")
            result = hierarchical_optimization(df_component, global_conflicts, license_bands)
        else:
            # Standard optimization path
            total_demand = df_component["bandwidth_mhz"].sum()
            band_capacity = GLOBAL_HIGH_FREQ - GLOBAL_LOW_FREQ
            
            if total_demand <= band_capacity:
                result = global_spectrum_optimization(df_component, global_conflicts)
                if result is not None:
                    optimized_chunks.append(result)
                    continue
            
            # Try cross-band coordination
            result_chunks = cross_band_coordination(df_component, license_bands)
            if result_chunks:
                optimized_chunks.extend(result_chunks)
            else:
                failed_components.append(i)
                logger.warning(f"⚠️ Conflict group {i+1} could not be optimized")
    
    if optimized_chunks:
        result_df = pd.concat(optimized_chunks, ignore_index=True)
        return result_df
    else:
        logger.error("⚠️ No feasible solutions found in any partition")
        return None


# --------------------------
# HIERARCHICAL OPTIMIZATION FOR LARGE COMPONENTS
# --------------------------

def hierarchical_optimization(df_component, global_conflicts, license_bands):
    """
    Handle very large conflict components by breaking them down hierarchically.
    This is a new function to address scalability concerns.
    """
    logger.info("Starting hierarchical optimization for large component")
    
    # Try to subdivide by clusters first
    clusters = df_component['cluster'].unique()
    
    if len(clusters) > 1:
        # Optimize each cluster separately first
        optimized_chunks = []
        
        for cluster_id in clusters:
            df_cluster = df_component[df_component['cluster'] == cluster_id].copy()
            state = cluster_id.split('_')[0]
            
            if state in license_bands:
                block_start = license_bands[state]["low_freq"]
                block_end = license_bands[state]["high_freq"]
                
                result = cp_optimize_cluster(df_cluster, block_start, block_end,
                                           cluster_id, global_conflicts)
                if result is not None:
                    optimized_chunks.append(result)
        
        if optimized_chunks:
            return pd.concat(optimized_chunks, ignore_index=True)
    
    # If cluster-based subdivision didn't work, try greedy assignment
    logger.info("Falling back to greedy frequency assignment for large component")
    return greedy_frequency_assignment(df_component, global_conflicts)


# --------------------------
# GREEDY FREQUENCY ASSIGNMENT (HEURISTIC)
# --------------------------

def greedy_frequency_assignment(df_component, conflicts):
    """
    Fast heuristic for very large problems where optimal solution is too slow.
    Assigns frequencies greedily based on bandwidth requirements.
    """
    logger.info("Using greedy heuristic for frequency assignment")
    
    # Sort stations by bandwidth (largest first)
    df_sorted = df_component.sort_values('bandwidth_mhz', ascending=False).copy()
    
    # Track assigned frequencies
    assigned_freqs = {}
    
    # Available channels
    channels = np.arange(GLOBAL_LOW_FREQ, GLOBAL_HIGH_FREQ, CHANNEL_STEP)
    
    for idx in df_sorted.index:
        bandwidth = df_sorted.loc[idx, 'bandwidth_mhz']
        slots_needed = int(np.ceil(bandwidth / CHANNEL_STEP))
        
        # Find conflicting stations that are already assigned
        conflicting_assigned = []
        for (i, j) in conflicts:
            if i == idx and j in assigned_freqs:
                conflicting_assigned.append(j)
            elif j == idx and i in assigned_freqs:
                conflicting_assigned.append(i)
        
        # Find available frequency range
        blocked_ranges = []
        for conf_idx in conflicting_assigned:
            start_idx, end_idx = assigned_freqs[conf_idx]
            blocked_ranges.append((start_idx, end_idx))
        
        # Find first available slot
        assigned = False
        for start_idx in range(len(channels) - slots_needed + 1):
            end_idx = start_idx + slots_needed
            
            # Check if this range conflicts with any blocked range
            conflict = False
            for blocked_start, blocked_end in blocked_ranges:
                if not (end_idx <= blocked_start or start_idx >= blocked_end):
                    conflict = True
                    break
            
            if not conflict:
                assigned_freqs[idx] = (start_idx, end_idx)
                assigned = True
                break
        
        if not assigned:
            logger.warning(f"Could not assign frequency to station {idx}")
    
    # Convert to result format
    assigned_start_freqs = []
    assigned_end_freqs = []
    
    for idx in df_component.index:
        if idx in assigned_freqs:
            start_idx, end_idx = assigned_freqs[idx]
            start_freq = channels[start_idx]
            end_freq = channels[min(end_idx, len(channels)-1)]
            assigned_start_freqs.append(start_freq)
            assigned_end_freqs.append(end_freq)
        else:
            assigned_start_freqs.append(np.nan)
            assigned_end_freqs.append(np.nan)
    
    df_result = df_component.copy()
    df_result["optimized_start_freq_mhz"] = assigned_start_freqs
    df_result["optimized_end_freq_mhz"] = assigned_end_freqs
    df_result["allocation_method"] = "Greedy_Heuristic"
    
    # Remove failed assignments
    df_result = df_result.dropna(subset=['optimized_start_freq_mhz'])
    
    return df_result


# --------------------------
# GLOBAL OPTIMIZATION WITH FIXED FREQUENCIES
# --------------------------

def global_spectrum_optimization(df, global_conflicts=None):
    """
    Global spectrum optimization with support for fixed frequency assignments.
    """
    logger.info("\n🌍 Starting global spectrum optimization...")
    
    # Build conflicts if not provided
    if global_conflicts is None:
        global_conflicts = build_global_conflict_matrix(df)
    
    # Available global spectrum channels
    global_channels = np.arange(GLOBAL_LOW_FREQ, GLOBAL_HIGH_FREQ, CHANNEL_STEP)
    global_channels = np.round(global_channels, 1)
    
    if len(global_channels) == 0:
        logger.error("❌ No global channels available")
        return None
    
    logger.info(f"📡 Global spectrum pool: {len(global_channels)} channels "
               f"({GLOBAL_LOW_FREQ}-{GLOBAL_HIGH_FREQ} MHz)")
    
    model = cp_model.CpModel()
    
    # Calculate slots required for each station
    slots_required = {}
    for idx in df.index:
        bandwidth = df.loc[idx, "bandwidth_mhz"]
        slots = int(np.ceil(bandwidth / CHANNEL_STEP))
        if slots > len(global_channels):
            logger.error(f"❌ Station {df.loc[idx, 'station_id']} requires {bandwidth}MHz "
                       f"but only {len(global_channels) * CHANNEL_STEP}MHz available")
            return None
        slots_required[idx] = slots
    
    # Channel assignment variables
    channel_vars = {}
    fixed_stations = []
    
    for idx in df.index:
        # Check if station has fixed frequency (if column exists and has value)
        has_fixed = False
        if 'fixed_start_freq' in df.columns:
            fixed_freq = df.loc[idx, 'fixed_start_freq']
            if not pd.isna(fixed_freq):
                has_fixed = True
                fixed_stations.append(idx)
                # Find the channel index for the fixed frequency
                fixed_idx = np.argmin(np.abs(global_channels - fixed_freq))
                var = model.NewConstant(fixed_idx)
                channel_vars[idx] = var
        
        if not has_fixed:
            max_start = len(global_channels) - slots_required[idx]
            if max_start < 0:
                logger.error(f"❌ Station {df.loc[idx, 'station_id']} requires more "
                           f"bandwidth than available")
                return None
            var = model.NewIntVar(0, max_start, f"global_chan_{idx}")
            channel_vars[idx] = var
    
    if fixed_stations:
        logger.info(f"Fixed frequency stations: {len(fixed_stations)}")
    
    # Global conflict constraints
    for (i_idx, j_idx) in global_conflicts:
        # Check if both indices are in our current dataframe
        if i_idx not in df.index or j_idx not in df.index:
            continue
            
        # Skip if both stations are fixed (nothing to optimize)
        if i_idx in fixed_stations and j_idx in fixed_stations:
            continue
            
        slots_i = slots_required[i_idx]
        slots_j = slots_required[j_idx]
        
        # Create ordering constraint
        bool_var = model.NewBoolVar(f"global_i_before_j_{i_idx}_{j_idx}")
        model.Add(channel_vars[i_idx] + slots_i <= channel_vars[j_idx]).OnlyEnforceIf(bool_var)
        model.Add(channel_vars[j_idx] + slots_j <= channel_vars[i_idx]).OnlyEnforceIf(bool_var.Not())
    
    # Ensure all stations fit within global spectrum
    for idx in df.index:
        if idx not in fixed_stations:
            model.Add(channel_vars[idx] + slots_required[idx] <= len(global_channels))
    
    # Objective: minimize total spectrum usage
    max_channel_var = model.NewIntVar(0, len(global_channels), "global_max_channel")
    for idx in df.index:
        model.Add(max_channel_var >= channel_vars[idx] + slots_required[idx])
    model.Minimize(max_channel_var)
    
    # Solve with configured timeout
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = SOLVER_TIMEOUT
    solver.parameters.num_search_workers = 8  # Use parallel search
    
    status = solver.Solve(model)
    
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        logger.error("❌ No feasible global solution found")
        return None
    
    # Extract results
    assigned_start_freqs = []
    assigned_end_freqs = []
    
    for idx in df.index:
        chan_idx = solver.Value(channel_vars[idx])
        start_freq = global_channels[chan_idx]
        slots = slots_required[idx]
        end_idx = chan_idx + slots
        end_freq = global_channels[min(end_idx, len(global_channels)-1)]
        assigned_start_freqs.append(start_freq)
        assigned_end_freqs.append(end_freq)
    
    df_result = df.copy()
    df_result["optimized_start_freq_mhz"] = assigned_start_freqs
    df_result["optimized_end_freq_mhz"] = assigned_end_freqs
    df_result["allocation_method"] = "Global_Optimization"
    
    spectrum_used = solver.Value(max_channel_var) * CHANNEL_STEP
    efficiency = (spectrum_used / (GLOBAL_HIGH_FREQ - GLOBAL_LOW_FREQ)) * 100
    
    logger.info(f"✅ Global optimization successful!")
    logger.info(f"📊 Total spectrum used: {spectrum_used} MHz ({efficiency:.1f}% of available)")
    logger.info(f"🔄 Frequency reuse across {len(df)} stations achieved")
    
    return df_result


# --------------------------
# CP OPTIMIZATION (LOCAL) WITH IMPROVEMENTS
# --------------------------

def cp_optimize_cluster(df_cluster, block_start, block_end, cluster_id,
                       global_conflicts=None):
    """
    Optimizes frequency allocation for a cluster using CP-SAT.
    Enhanced to consider cross-cluster conflicts if provided.
    """
    LEGAL_CHANNELS = np.arange(block_start, block_end, CHANNEL_STEP)
    LEGAL_CHANNELS = np.round(LEGAL_CHANNELS, 1)
    
    if len(LEGAL_CHANNELS) == 0:
        logger.warning(f"⚠️ No legal channels for {cluster_id}")
        return None
    
    model = cp_model.CpModel()
    
    slots_required = {
        idx: int(np.ceil(df_cluster.loc[idx, "bandwidth_mhz"] / CHANNEL_STEP))
        for idx in df_cluster.index
    }
    
    channel_vars = {}
    for idx in df_cluster.index:
        max_start = len(LEGAL_CHANNELS) - slots_required[idx]
        if max_start < 0:
            logger.warning(f"Station {idx} requires more bandwidth than available in local band")
            return None
        var = model.NewIntVar(0, max_start, f"chan_idx_{idx}")
        channel_vars[idx] = var
    
    # Build local conflicts
    if global_conflicts:
        # Extract relevant conflicts from global set
        conflicts = []
        cluster_indices = set(df_cluster.index)
        for (i, j) in global_conflicts:
            if i in cluster_indices and j in cluster_indices:
                conflicts.append((i, j))
    else:
        # Build conflicts locally
        conflicts = build_conflict_pairs(df_cluster)
    
    # Add conflict constraints
    for i_idx, j_idx in conflicts:
        slots_i = slots_required[i_idx]
        slots_j = slots_required[j_idx]
        bool_var = model.NewBoolVar(f"i_before_j_{i_idx}_{j_idx}")
        model.Add(channel_vars[i_idx] + slots_i <= channel_vars[j_idx]).OnlyEnforceIf(bool_var)
        model.Add(channel_vars[j_idx] + slots_j <= channel_vars[i_idx]).OnlyEnforceIf(bool_var.Not())
    
    for idx in df_cluster.index:
        model.Add(channel_vars[idx] + slots_required[idx] <= len(LEGAL_CHANNELS))
    
    # Objective: minimize the highest used channel
    max_channel_var = model.NewIntVar(0, len(LEGAL_CHANNELS), "max_channel")
    for idx in df_cluster.index:
        model.Add(max_channel_var >= channel_vars[idx] + slots_required[idx])
    model.Minimize(max_channel_var)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = SOLVER_TIMEOUT
    status = solver.Solve(model)
    
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        logger.warning(f"❌ No feasible solution for cluster {cluster_id}")
        return None
    
    assigned_start_freqs = []
    assigned_end_freqs = []
    for idx in df_cluster.index:
        chan_idx = solver.Value(channel_vars[idx])
        start_freq = LEGAL_CHANNELS[chan_idx]
        slots = slots_required[idx]
        end_idx = chan_idx + slots
        end_freq = LEGAL_CHANNELS[min(end_idx, len(LEGAL_CHANNELS)-1)]
        assigned_start_freqs.append(start_freq)
        assigned_end_freqs.append(end_freq)
    
    df_cluster["optimized_start_freq_mhz"] = assigned_start_freqs
    df_cluster["optimized_end_freq_mhz"] = assigned_end_freqs
    df_cluster["allocation_method"] = "Local_CP"
    
    logger.info(f"✅ Cluster {cluster_id} optimized with local CP")
    return df_cluster


def build_conflict_pairs(df_cluster):
    """
    Returns list of station index pairs that cannot overlap.
    Legacy function kept for compatibility.
    """
    # Convert to dict format and extract just the pairs
    conflicts_dict = build_global_conflict_matrix(df_cluster, use_spatial_index=False)
    return list(conflicts_dict.keys())


# --------------------------
# CROSS-BAND COORDINATION
# --------------------------

def cross_band_coordination(df, license_bands=None):
    """
    Implements dynamic band allocation using cross-band coordination.
    Enhanced to properly handle cross-cluster conflicts.
    """
    logger.info("\n🔗 Performing cross-band coordination...")
    
    if license_bands is None:
        license_bands = load_license_bands()
    
    # First, check for cross-cluster conflicts
    global_conflicts = build_global_conflict_matrix(df)
    
    # Build a graph to check if clusters interfere
    cluster_graph = nx.Graph()
    clusters = df['cluster'].unique()
    cluster_graph.add_nodes_from(clusters)
    
    # Check if any stations from different clusters conflict
    for (i, j) in global_conflicts:
        cluster_i = df.loc[i, 'cluster']
        cluster_j = df.loc[j, 'cluster']
        if cluster_i != cluster_j:
            cluster_graph.add_edge(cluster_i, cluster_j)
    
    # Find connected components of clusters
    cluster_components = list(nx.connected_components(cluster_graph))
    logger.info(f"Found {len(cluster_components)} independent cluster groups")
    
    optimized_chunks = []
    failed_clusters = []
    
    # Process each cluster component
    for component_clusters in cluster_components:
        component_df = df[df['cluster'].isin(component_clusters)].copy()
        
        if len(component_clusters) == 1:
            # Single cluster with no external conflicts - can use local optimization
            cluster_id = list(component_clusters)[0]
            state = cluster_id.split("_")[0]
            
            if state not in license_bands:
                logger.warning(f"⚠️ Unknown state {state} for cluster {cluster_id}")
                failed_clusters.append(cluster_id)
                continue
            
            block_start = license_bands[state]["low_freq"]
            block_end = license_bands[state]["high_freq"]
            df_cluster = component_df.copy()
            total_demand = df_cluster["bandwidth_mhz"].sum()
            band_capacity = block_end - block_start
            
            logger.info(f"Cluster {cluster_id}: demand={total_demand}MHz, "
                       f"capacity={band_capacity}MHz")
            
            if total_demand <= band_capacity:
                result = cp_optimize_cluster(df_cluster, block_start, block_end,
                                           cluster_id, global_conflicts)
                if result is not None:
                    optimized_chunks.append(result)
                    continue
            
            failed_clusters.extend(component_clusters)
        else:
            # Multiple interfering clusters - need global optimization
            logger.info(f"Clusters {component_clusters} interfere - using global optimization")
            result = global_spectrum_optimization(component_df, global_conflicts)
            if result is not None:
                optimized_chunks.append(result)
            else:
                failed_clusters.extend(component_clusters)
    
    # Second pass: global optimization for failed clusters
    if failed_clusters:
        failed_df = df[df["cluster"].isin(failed_clusters)].copy()
        logger.info(f"🔄 {len(failed_clusters)} clusters will use global spectrum pool")
        global_result = global_spectrum_optimization(failed_df, global_conflicts)
        if global_result is not None:
            optimized_chunks.append(global_result)
    
    return optimized_chunks


# --------------------------
# ENHANCED PLOTTING
# --------------------------

def plot_cluster(df_cluster, cluster_id, block_start=None, block_end=None, save_path=None):
    """
    Enhanced plotting with better visualization and conflict indication.
    """
    fig, ax = plt.subplots(figsize=(14, max(0.5 * len(df_cluster), 4)))
    
    # Color by allocation method
    method_colors = {
        'Local_CP': 'lightblue',
        'Global_Optimization': 'lightgreen',
        'Greedy_Heuristic': 'lightyellow'
    }
    
    # Sort by start frequency for better visualization
    df_plot = df_cluster.sort_values('optimized_start_freq_mhz')
    
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        width = row.optimized_end_freq_mhz - row.optimized_start_freq_mhz
        color = method_colors.get(row.allocation_method, 'lightgray')
        
        ax.barh(
            y=i,
            width=width,
            left=row.optimized_start_freq_mhz,
            color=color,
            edgecolor="black",
            alpha=0.8,
            linewidth=1.5
        )
        
        # Enhanced label with more info
        label = f"{row.station_id}\n{int(width)}MHz\n{row.area_type}"
        ax.text(
            row.optimized_start_freq_mhz + width/2,
            i,
            label,
            va="center",
            ha="center",
            fontsize=8,
            weight='bold'
        )
    
    # Title and labels
    title = f"Spectrum Allocation - Cluster {cluster_id}"
    if block_start and block_end:
        title += f"\nLicense Band: {block_start}-{block_end} MHz"
        # Add license band boundaries
        ax.axvline(x=block_start, color='red', linestyle='--', alpha=0.5, label='License Band')
        ax.axvline(x=block_end, color='red', linestyle='--', alpha=0.5)
    
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel("Frequency (MHz)", fontsize=12)
    ax.set_ylabel("Stations", fontsize=12)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels([f"Station {i+1}" for i in range(len(df_plot))])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.8)
               for method, color in method_colors.items()
               if method in df_cluster['allocation_method'].values]
    labels = [method for method in method_colors.keys()
              if method in df_cluster['allocation_method'].values]
    ax.legend(handles, labels, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        filename = save_path
    else:
        filename = f"cluster_{cluster_id}_allocation.png"
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Plot saved: {filename}")


# --------------------------
# ENHANCED METRICS WITH TIMING
# --------------------------

def analyze_optimization_metrics(result_df, start_time=None):
    """
    Comprehensive optimization metrics with performance timing.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE OPTIMIZATION METRICS")
    print("=" * 80)
    
    # Timing information
    if start_time:
        total_time = time.time() - start_time
        print(f"\n⏱️  Total optimization time: {total_time:.2f} seconds")
    
    total_stations = len(result_df)
    print(f"\n📊 Total stations assigned frequencies: {total_stations}")
    
    # Method analysis
    method_counts = result_df["allocation_method"].value_counts()
    print(f"\n🔧 Allocation methods used:")
    for method, count in method_counts.items():
        percentage = (count / total_stations) * 100
        print(f"   • {method}: {count} stations ({percentage:.1f}%)")
    
    # Spectrum usage analysis
    min_freq = result_df["optimized_start_freq_mhz"].min()
    max_freq = result_df["optimized_end_freq_mhz"].max()
    total_spectrum_used = max_freq - min_freq
    total_bandwidth = result_df["bandwidth_mhz"].sum()
    
    print(f"\n📡 Spectrum Usage Analysis:")
    print(f"   • Frequency range: {min_freq:.1f} MHz - {max_freq:.1f} MHz")
    print(f"   • Total spectrum span: {total_spectrum_used:.1f} MHz")
    print(f"   • Total bandwidth demand: {total_bandwidth:.1f} MHz")
    print(f"   • Available spectrum: {GLOBAL_HIGH_FREQ - GLOBAL_LOW_FREQ} MHz")
    
    if total_spectrum_used > 0:
        efficiency = (total_bandwidth / total_spectrum_used) * 100
        print(f"   • Spectrum efficiency: {efficiency:.1f}%")
        
        # Compare to theoretical minimum
        theoretical_min = total_bandwidth
        overhead = ((total_spectrum_used - theoretical_min) / theoretical_min) * 100
        print(f"   • Overhead vs theoretical minimum: {overhead:.1f}%")
    
    # State/region analysis
    state_stats = result_df.groupby("state").agg({
        "bandwidth_mhz": ["count", "sum"],
        "optimized_start_freq_mhz": "min",
        "optimized_end_freq_mhz": "max"
    }).round(1)
    
    print(f"\n🗺️  State-level Analysis:")
    for state in state_stats.index[:5]:  # Show top 5 states
        stats = state_stats.loc[state]
        station_count = stats[("bandwidth_mhz", "count")]
        total_bw = stats[("bandwidth_mhz", "sum")]
        freq_span = stats[("optimized_end_freq_mhz", "max")] - \
                   stats[("optimized_start_freq_mhz", "min")]
        print(f"   • {state}: {station_count} stations, {total_bw}MHz demand, "
              f"{freq_span}MHz span")
    
    if len(state_stats) > 5:
        print(f"   • ... and {len(state_stats) - 5} more states")
    
    # Cluster analysis
    cluster_stats = result_df.groupby("cluster").agg({
        "bandwidth_mhz": ["count", "sum"],
        "optimized_start_freq_mhz": "min",
        "optimized_end_freq_mhz": "max",
        "allocation_method": lambda x: x.value_counts().index[0]  # Most common method
    }).round(1)
    
    print(f"\n🏘️  Cluster Analysis (showing first 5):")
    for cluster_id in list(cluster_stats.index)[:5]:
        stats = cluster_stats.loc[cluster_id]
        station_count = stats[("bandwidth_mhz", "count")]
        total_bw = stats[("bandwidth_mhz", "sum")]
        freq_span = stats[("optimized_end_freq_mhz", "max")] - \
                   stats[("optimized_start_freq_mhz", "min")]
        method = stats[("allocation_method", "<lambda>")]
        print(f"   • {cluster_id}: {station_count} stations, {total_bw}MHz demand, "
              f"{freq_span}MHz span, {method}")
    
    # Frequency reuse analysis
    print(f"\n🔄 Frequency Reuse Analysis:")
    
    # Count actual frequency overlaps
    freq_overlaps = 0
    reuse_opportunities = 0
    
    # Build conflict set for quick lookup
    conflict_set = set()
    for i, row1 in result_df.iterrows():
        for j, row2 in result_df.iterrows():
            if j <= i:
                continue
            
            # Check if frequencies overlap
            if (row1["optimized_start_freq_mhz"] < row2["optimized_end_freq_mhz"] and
                row2["optimized_start_freq_mhz"] < row1["optimized_end_freq_mhz"]):
                freq_overlaps += 1
                
                # Verify these stations don't conflict
                distance_km = haversine_distance(
                    row1["y_coord"], row1["x_coord"],
                    row2["y_coord"], row2["x_coord"]
                )
                
                threshold = min(
                    AREA_DISTANCE_THRESHOLDS.get(row1["area_type"], 50),
                    AREA_DISTANCE_THRESHOLDS.get(row2["area_type"], 50)
                )
                
                if distance_km >= threshold or not check_antenna_interference(row1, row2):
                    reuse_opportunities += 1
    
    print(f"   • Frequency overlaps detected: {freq_overlaps}")
    print(f"   • Valid reuse opportunities: {reuse_opportunities}")
    
    if freq_overlaps > 0:
        reuse_rate = (reuse_opportunities / freq_overlaps) * 100
        print(f"   • Reuse validation rate: {reuse_rate:.1f}%")
    
    # Area type distribution
    area_dist = result_df['area_type'].value_counts()
    print(f"\n🏙️  Area Type Distribution:")
    for area_type, count in area_dist.items():
        percentage = (count / total_stations) * 100
        print(f"   • {area_type}: {count} stations ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)


# --------------------------
# MAIN RUNNER FUNCTION
# --------------------------

def run_optimizer(input_csv, output_csv, use_partitioning=True,
                 license_bands_file=None, plot_results=True):
    """
    Main optimization runner with all improvements integrated.
    """
    start_time = time.time()
    
    # Output tool summary
    output_tool_summary()
    
    logger.info("🚀 Starting enhanced spectrum optimizer...")
    
    # Load data
    try:
        df = load_data(input_csv)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None
    
    # Load license bands configuration
    license_bands = load_license_bands(license_bands_file)
    
    # Run optimization
    if use_partitioning:
        logger.info("\n🌟 Using Partitioned Optimization Strategy")
        result = partition_and_optimize(df, license_bands)
    else:
        logger.info("\n🌍 Using Direct Global Optimization")
        result = global_spectrum_optimization(df)
    
    if result is not None:
        # Ensure required columns exist
        required_cols = [
            "station_id", "x_coord", "y_coord",
            "bandwidth_mhz", "azimuth_deg",
            "state", "cluster", "area_type",
            "optimized_start_freq_mhz",
            "optimized_end_freq_mhz",
            "allocation_method"
        ]
        
        # Add any missing columns
        for col in required_cols:
            if col not in result.columns and col in df.columns:
                result[col] = df[col]
        
        # Sort by frequency for better visualization
        result = result.sort_values("optimized_start_freq_mhz")
        
        # Save results
        result.to_csv(output_csv, index=False)
        logger.info(f"✅ Optimized results written to: {output_csv}")
        
        # Generate plots if requested
        if plot_results:
            logger.info("\n📊 Generating visualization plots...")
            for cluster_id in result["cluster"].unique():
                cluster_data = result[result["cluster"] == cluster_id]
                state = cluster_id.split("_")[0]
                
                if state in license_bands and \
                   cluster_data["allocation_method"].iloc[0] == "Local_CP":
                    plot_cluster(cluster_data, cluster_id,
                               license_bands[state]["low_freq"],
                               license_bands[state]["high_freq"])
                else:
                    plot_cluster(cluster_data, cluster_id)
        
        # Print comprehensive metrics
        analyze_optimization_metrics(result, start_time)
        
        logger.info(f"\n🎉 Optimization complete! Results saved to {output_csv}")
        return result
    else:
        logger.error("\n❌ Optimization failed. Please review input data and constraints.")
        return None


# --------------------------
# MAIN EXECUTION
# --------------------------

if __name__ == "__main__":
    # Configuration
    input_csv = "test_spectrum_dataset.csv"
    output_csv = "optimized_spectrum_improved.csv"
    license_bands_file = None  # Set to path if you have a JSON file with license bands
    
    # Run the improved optimizer
    result = run_optimizer(
        input_csv=input_csv,
        output_csv=output_csv,
        use_partitioning=True,  # Recommended for scalability
        license_bands_file=license_bands_file,
        plot_results=True
    )
    
    if result is not None:
        print(f"\n✅ Successfully optimized {len(result)} stations")
        print(f"📁 Results saved to: {output_csv}")
    else:
        print("\n❌ Optimization failed - check the logs above for details")
