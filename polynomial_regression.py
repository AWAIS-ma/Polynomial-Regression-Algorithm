
#    POLYNOMIAL REGRESSION: ENERGY USAGE Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. DATASET IMPORT
data = pd.read_csv("D:\Machine Learning Alogrithms\polynomial_regression\Polynomial-Regression-Algorithm\energy_data.csv")


print("\n----- DATASET LOADED -----")
print(data.head())

# 2. DEFINE INPUT & TARGET

# Input features: Temperature, Humidity, Wind_Speed
X = data[["Temperature", "Humidity", "Wind_Speed"]].values

# Target: Electricity Usage
y = data["Electricity_Usage"].values

# 3. POLYNOMIAL TRANSFORMATION

degree = 2  # Polynomial degree
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# 4. LINEAR REGRESSION MODEL ON POLYNOMIAL FEATURES
model = LinearRegression()
model.fit(X_poly, y)

print("\nPolynomial Regression Model Ready")

# 5. VISUALIZATION (Temperature vs Electricity Usage)

plt.figure(figsize=(8,5))
plt.scatter(data["Temperature"], y, color='blue', label="Actual Usage")

# Predicted values for the scatter points
y_pred = model.predict(poly.transform(X))
plt.plot(data["Temperature"], y_pred, color='red', label=f"Polynomial Regression (deg={degree})")

plt.xlabel("Temperature (°C)")
plt.ylabel("Electricity Usage (kWh)")
plt.title("Polynomial Regression: Temperature vs Electricity Usage")
plt.legend()
plt.show()

# ---------------------------
# 6. USER INPUT PREDICTION
# ---------------------------
print("\n----- USER INPUT PREDICTION -----")

user_temp = float(input("Enter Temperature (°C): "))
user_humidity = float(input("Enter Humidity (%): "))
user_wind = float(input("Enter Wind Speed (km/h): "))

user_input_poly = poly.transform([[user_temp, user_humidity, user_wind]])
predicted_usage = model.predict(user_input_poly)

print(f"\nPredicted Electricity Usage: {predicted_usage[0]:.2f} kWh")
