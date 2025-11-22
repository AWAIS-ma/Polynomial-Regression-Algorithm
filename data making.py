import pandas as pd
import numpy as np

# 1. DEFINE DATA RANGES
max_entries = 300
temperatures = np.linspace(10, 40, max_entries)  # 10°C se 40°C 
humidity = np.linspace(85, 98, max_entries)      # Humidity % 85-98
wind_speed = np.linspace(15, 21, max_entries)   # Wind Speed 15-21 km/h

# 2. SIMULATE NON-LINEAR ELECTRICITY USAGE
# -----------------------------
# Usage decreases initially, then increases (U-shape curve)
electricity_usage = 0.5*(temperatures-20)**2 + 50  # quadratic pattern

# 3. CREATE DATAFRAME
data = pd.DataFrame({
    "Temperature": temperatures,
    "Humidity": humidity.astype(int),
    "Wind_Speed": wind_speed.astype(int),
    "Electricity_Usage": electricity_usage.astype(int)
})

# 4. SAVE CSV
data.to_csv("energy_data.csv", index=False)
print("Dataset saved as 'energy_data.csv'\n")
print(data)
