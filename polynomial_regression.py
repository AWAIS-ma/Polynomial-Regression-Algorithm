# POLYNOMIAL REGRESSION: ENERGY USAGE Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. DATASET IMPORT
data = pd.read_csv(r"energy_data.csv")  
print("\n----- DATASET LOADED -----")

# 2. DEFINE INPUT & TARGET
X = data[["Temperature", "Humidity", "Wind_Speed"]].values
y = data["Electricity_Usage"].values

# 3. POLYNOMIAL TRANSFORMATION
degree = 2
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# 4. TRAIN-TEST SPLIT (IMPORTANT FOR TRUE EVALUATION)
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

# 5. MODEL TRAINING
model = LinearRegression()
model.fit(X_train, y_train)
print("\nPolynomial Regression Model Ready")

# 6. MODEL EVALUATION ON TEST DATA
y_test_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)
n = X_test.shape[0]
p = X_test.shape[1] - 1
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

print("\n----- TEST SET EVALUATION -----")
print(f"MSE        : {mse:.4f}")
print(f"RMSE       : {rmse:.4f}")
print(f"R² Score   : {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")

# 7. VISUALIZATION (Temperature vs Electricity Usage)
y_full_pred = model.predict(X_poly)
plt.figure(figsize=(8,5))
plt.scatter(data["Temperature"], y, label="Actual Usage")
plt.plot(data["Temperature"], y_full_pred, color='red', label=f"Polynomial Regression (deg={degree})")
plt.xlabel("Temperature (°C)")
plt.ylabel("Electricity Usage (kWh)")
plt.title("Polynomial Regression: Temperature vs Electricity Usage")
plt.legend()
plt.show()

# 8. USER INPUT PREDICTION
print("\n----- USER INPUT PREDICTION -----")
user_temp = float(input("Enter Temperature (°C): "))
user_humidity = float(input("Enter Humidity (%): "))
user_wind = float(input("Enter Wind Speed (km/h): "))

user_input_poly = poly.transform([[user_temp, user_humidity, user_wind]])
predicted_usage = model.predict(user_input_poly)
print(f"\nPredicted Electricity Usage: {predicted_usage[0]:.2f} kWh")
