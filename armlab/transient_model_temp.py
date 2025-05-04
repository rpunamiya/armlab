import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc
from scipy.signal import savgol_filter

# ======================
# 1. Load and Inspect Data
# ======================
file_path = r"C:\Users\rpuna\OneDrive - Stanford\Research\ARMLab\Temperature\hollow_steel1.csv"
file_path2 = r"C:\Users\rpuna\OneDrive - Stanford\Research\ARMLab\Temperature\rubber3_1.csv"
df = pd.read_csv(file_path, header=None)
df2 = pd.read_csv(file_path2, header=None)
df.columns = ["time", "heat_flux", "temp_flux_sensor", "unused", "temp_thermocouple"]
df2.columns = ["time", "heat_flux", "temp_flux_sensor", "unused", "temp_thermocouple"]

# ======================
# 2. Define Known Parameters
# ======================
# Sensor spacing between surface (flux sensor) and interior thermocouple (in meters)
x = 0.043  # for example, 1 cm

# Average heat flux (if you assume a constant flux condition)
# You might choose to average over a time window where the flux is stable.
q_avg = 482.97 # W/m^2
q2_avg = 377.30 # W/m^2

# Initial temperature from the interior sensor; assume the system starts at a uniform T0.
T0 = df["temp_thermocouple"].iloc[0]
T0_2 = df2["temp_thermocouple"].iloc[0]

# ======================
# 3. Define the Transient Conduction Model Function
# ======================
def transient_model(t, k, alpha):
    """
    Transient conduction model for a semi-infinite solid with constant surface heat flux.
    
    Parameters:
      t     : time (s)
      k     : thermal conductivity (W/m·K)
      alpha : thermal diffusivity (m^2/s)
      
    Returns:
      Temperature at distance x and time t.
    """
    # Avoid division by zero at t=0 by replacing any t==0 with a small number
    t = np.where(t == 0, 1e-6, t)
    return T0 + (2 * q_avg / k) * np.sqrt(alpha * t / np.pi) * np.exp(-x**2 / (4 * alpha * t)) - (q_avg * x / k) * erfc(x / (2 * np.sqrt(alpha * t)))

# ======================
# 4. Choose a Time Window for Fitting
# ======================
# Choose a window where the transient response is clearly captured.
# Adjust these values based on your plot of the temperature data.
mask = (df["time"] >= 15) & (df["time"] <= 275)
mask2 = (df2["time"] >= 15) & (df2["time"] <= 275)
t_fit = df["time"][mask].to_numpy()
t_fit2 = df2["time"][mask2].to_numpy()
T_fit = df["temp_thermocouple"][mask].to_numpy()
T_fit2 = df2["temp_thermocouple"][mask2].to_numpy()

# Smooth the temperature data using a Savitzky–Golay filter
# window_length must be odd and chosen based on the number of points in t_fit
window_length = 20  # adjust as needed (should be less than len(t_fit))
polyorder = 3       # polynomial order for the filter
T_fit_smooth = savgol_filter(T_fit, window_length=window_length, polyorder=polyorder)
T_fit2_smooth = savgol_filter(T_fit2, window_length=window_length, polyorder=polyorder)

# plt.figure(figsize=(10, 6))
# plt.plot(t_fit, T_fit, 'o', label="Interior Temperature Data")
# plt.plot(t_fit, T_fit_smooth, 'r-', label="Smoothed Data")
# plt.xlabel("Time (s)")
# plt.ylabel("Temperature (°C)")
# plt.title("Interior Temperature vs Time")
# plt.legend()
# plt.show()

# ======================
# 5. Fit the Model to the Data
# ======================
# Provide initial guesses for k and alpha. Adjust these guesses based on your expectations.
initial_guess = [0.2, 9e-8]  # example: k in W/m·K, alpha in m^2/s

params, covariance = curve_fit(transient_model, t_fit, T_fit_smooth, p0=initial_guess, maxfev=1000000)
k_fit, alpha_fit = params

print("Fitted thermal conductivity (W/m·K):", k_fit)
print("Fitted thermal diffusivity (m^2/s):", alpha_fit)

# ======================
# 6. Plot the Fitted Curve Against the Data
# ======================
t_model = np.linspace(t_fit[0], t_fit[-1], 200)
t2_model = np.linspace(t_fit2[0], t_fit2[-1], 200)
T_model = transient_model(t_model, k_fit, alpha_fit)
T_model2 = transient_model(t2_model, k_fit, alpha_fit)

plt.figure(figsize=(10, 6))
plt.plot(t_fit, T_fit_smooth, '-', label="Side Temperature Data (Steel)", color='blue')
plt.plot(t_model, T_model, '-', label="Fitted Model (Steel)", color='green')
plt.plot(t_fit2, T_fit2_smooth, '-', label="Side Temperature Data (Rubber)", color='red')
plt.plot(t2_model, T_model2, '-', label="Fitted Model (Rubber)", color='orange')
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Fitting Transient Conduction Model")
plt.legend()
plt.show()
