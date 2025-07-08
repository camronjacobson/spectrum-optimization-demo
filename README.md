# Spectrum Optimization Tool

This project helps assign radio spectrum frequencies to radio stations in a way that minimizes total spectrum use and avoids interference.

It includes:

- A **Spectrum Optimizer** → assigns frequencies to stations under distance and antenna constraints.
- A **Result Analyzer** → checks and visualizes how spectrum was allocated.

This repository is private and intended for internal use only.

---

## How to Set Up

This project uses an Anaconda environment for easy installation.

---

### 1. Install Anaconda (if needed)

Download and install Anaconda:

[Download Anaconda here.](https://www.anaconda.com/products/distribution)

---

### 2. Clone the Repository

If you haven’t already:

```bash
git clone <your-private-repo-url>
cd spectrum-optimization-demo
```

---

### 3. Create the Environment

Run the following command in the repo folder:

```bash
conda env create -f environment.yml
```

---

### 4. Activate the Environment

Activate the environment:

```bash
conda activate spectrum-opt-env
```

Now you’re ready to run the optimizer and analyzer scripts!

---

### Updating the Environment

If `environment.yml` changes, update your environment with:

```bash
conda env update -f environment.yml --prune
```

---

## How to Run the Optimizer

1. **Prepare Your Input CSV**

Create a CSV file (e.g. `test_spectrum_dataset.csv`) with columns like:

| station_id | x_coord | y_coord | bandwidth_mhz | azimuth_deg | cluster |
|------------|---------|---------|---------------|-------------|---------|
| STA001     | -118.3  | 34.1    | 5             | 90          | CA_0    |
| ...        | ...     | ...     | ...           | ...         | ...     |

Optional columns:

- `state`
- `area_type` (urban, suburban, rural)
- `elevation_deg`
- `fixed_start_freq`, `fixed_end_freq`

If not present, the optimizer fills in defaults automatically.

---

2. **Run the Optimizer Script**

From the terminal:

```bash
python spectrum_optimizer.py
```

This will:

- Run the optimization
- Create an output CSV file (e.g. `optimized_spectrum_improved.csv`)
- Save plots of frequency allocations for each cluster

---

## How to Analyze Results

Once optimization is done, you can analyze the output using:

```bash
python Spectrum_Optimizer_Result_Analyzer.py
```

Or use it in Python like this:

```python
from Spectrum_Optimizer_Result_Analyzer import (
    analyze_optimization_results,
    visualize_spectrum_allocation,
    find_optimization_issues
)

# Analyze results
results_df = analyze_optimization_results("optimized_spectrum_improved.csv")

# Plot allocations
visualize_spectrum_allocation(results_df)

# Check for issues
find_optimization_issues(results_df)
```

---

## What You’ll Get

 A CSV with all stations and their assigned frequencies  
 Plots showing how spectrum is divided among stations  
 Summaries of spectrum usage and efficiency  
 Warnings if stations might still overlap

---

## Notes

- For real regulatory data, replace the dummy license band settings in the code with actual values.
- If your dataset is large, the tool will automatically use faster methods for conflict checking.

---


