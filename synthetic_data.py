import matplotlib
matplotlib.use("TkAgg")  # Required for macOS GUI rendering
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from ortools.sat.python import cp_model

# 1. Generate Synthetic Data
#def generate_synthetic_stations(n=100, seed=42, outfile="spectrum_data.csv"):
#    random.seed(seed)
#    data = []
#    for i in range(n):
 #       station = {
   #         "station_id": f"S{i+1}",
 #           "x_coord": random.uniform(0, 100),
   #         "y_coord": random.uniform(0, 100),
   #         "bandwidth": random.choice([1, 2, 3, 4]),
   #         "azimuth": random.uniform(0, 360)
   #     }
  #      data.append(station)
 #   df = pd.DataFrame(data)
 #   df.to_csv(outfile, index=False)
 #   print(f"✅ Generated {n} synthetic stations in '{outfile}'")

# 2. Load CSV Data
def load_stations_from_csv(filepath):
    df = pd.read_csv(filepath)
    stations = []
    for _, row in df.iterrows():
        stations.append({
            "id": row["station_id"],
            "x": row["x_coord"],
            "y": row["y_coord"],
            "bw": int(row["bandwidth"]),
            "dir": row["azimuth"]
        })
    return stations

# 3. OR-Tools Optimization
def optimize_spectrum(stations):
    model = cp_model.CpModel()
    n = len(stations)
    max_channel = sum(s["bw"] for s in stations)

    start_vars = []
    for i in range(n):
        var = model.NewIntVar(0, max_channel - stations[i]["bw"], f"start_{i}")
        start_vars.append(var)

    intervals = []
    for i in range(n):
        interval = model.NewIntervalVar(start_vars[i],
                                        stations[i]["bw"],
                                        start_vars[i] + stations[i]["bw"],
                                        f"intv_{i}")
        intervals.append(interval)

    # Add no-overlap constraint
    model.AddNoOverlap(intervals)

    # Objective: minimize the max used channel
    max_used = model.NewIntVar(0, max_channel, "max_used")
    for i in range(n):
        model.Add(start_vars[i] + stations[i]["bw"] <= max_used)
    model.Minimize(max_used)

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assignments = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"✅ Solution found with max channel = {solver.Value(max_used)}")
        for i in range(n):
            start = solver.Value(start_vars[i])
            print(f"Station {stations[i]['id']}: start_channel = {start} (bw = {stations[i]['bw']})")
            assignments.append({
                "id": stations[i]["id"],
                "start_channel": start,
                "bandwidth": stations[i]["bw"],
                "x": stations[i]["x"],
                "y": stations[i]["y"],
                "dir": stations[i]["dir"]
            })
    else:
        print("❌ No feasible solution found.")

    return assignments

# 4. Main Flow
if __name__ == "__main__":
  #  generate_synthetic_stations(n=1000)  # Uncomment if you want fresh data
    stations = load_stations_from_csv("spectrum_data.csv")
    assignments = optimize_spectrum(stations)

    if not assignments:
        print("⚠️ No assignments to plot.")
        exit()

    # 5. Export Results
    df_out = pd.DataFrame(assignments)
    df_out.to_csv("frequency_assignments.csv", index=False)

# 6. Plot Results
import math

n_assignments = len(assignments)
fig_height = max(6, min(0.15 * n_assignments, 50))  # Dynamically scale height up to 50
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_height))

# Frequency Assignment Timeline
for i, s in enumerate(assignments):
    ax1.barh(i, s["bandwidth"], left=s["start_channel"], color=plt.cm.tab20(i % 20))
    if n_assignments <= 100:  # Only label if not too crowded
        ax1.text(s["start_channel"] + s["bandwidth"] / 2, i, s["id"],
                 ha="center", va="center", color="white", fontsize=6)

if n_assignments <= 100:
    ax1.set_yticks(range(n_assignments))
    ax1.set_yticklabels([s["id"] for s in assignments])
else:
    ax1.set_yticks([])

ax1.set_xlabel("Frequency Channel")
ax1.set_title(f"Frequency Assignment Timeline (n={n_assignments})")
ax1.set_ylim(-1, n_assignments)

# Antenna Direction Map
for s in assignments:
    x, y = s["x"], s["y"]
    angle = np.radians(s["dir"])
    dx = np.cos(angle)
    dy = np.sin(angle)
    ax2.plot(x, y, 'bo')
    ax2.arrow(x, y, dx * 5, dy * 5, head_width=1.5, head_length=2,
              fc='blue', ec='blue')
    if n_assignments <= 200:
        ax2.text(x + 1, y + 1, s["id"], fontsize=6)

ax2.set_xlabel("X Coordinate (km)")
ax2.set_ylabel("Y Coordinate (km)")
ax2.set_title("Antenna Direction Map")

plt.tight_layout()
# plt.savefig("spectrum_visuals.png")
plt.show()

