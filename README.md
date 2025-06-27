# Spectrum Optimization Demo

This repository demonstrates a Python tool for optimizing radio frequency assignments for stations, using Google's OR-Tools.

---

## 📂 Included Files

### spectrum_data.py

- The main Python script for running the optimization.
- Loads station data from the CSV file.
- Sets up constraints to avoid interference between stations.
- Solves the optimization problem using OR-Tools.
- Prints results and optionally saves outputs.

---

### spectrum_data.csv

- Sample input data for testing the optimization tool.
- Contains information for multiple radio stations, such as:
  - Station ID
  - Coordinates (latitude, longitude)
  - Azimuth
  - Elevation
  - Bandwidth
- You can replace this file with your own data for different test scenarios.

---

## ▶️ How to Run

1. Install required Python libraries:

    ```bash
    pip install ortools pandas matplotlib
    ```

2. Run the optimization script:

    ```bash
    python spectrum_data.py
    ```

3. Check console output and any saved output files to see the frequency assignments.

---

## 🔒 Note

This repository is private and intended for internal demonstration purposes only.
