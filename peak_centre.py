import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load data from CSV file with specified column names
data = pd.read_csv('chemistry/acetone_water/water.CSV', names=['Wavenumbers', 'Intensities'])

# Extract wavenumbers and intensities
wavenumbers = data['Wavenumbers'].values
intensities = data['Intensities'].values


# Define Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)


# Define function to fit Gaussian to data
def fit_gaussian(x, y):
    # Initial guess for parameters (amplitude, mean, stddev)
    p0 = [max(y), np.mean(x), np.std(x)]

    # Set bounds for the parameters
    lower_bounds = [0, 3000, 1]  # Example lower bounds for amplitude, mean (center), and stddev
    upper_bounds = [np.inf, 3700, 100]  # Example upper bounds

    # Fit Gaussian function to data
    popt, _ = curve_fit(gaussian, x, y, p0=p0, bounds=(lower_bounds, upper_bounds))

    return popt  # Return optimized parameters


# Fit Gaussian to the entire spectrum
popt = fit_gaussian(wavenumbers, intensities)
fit_curve = gaussian(wavenumbers, *popt)

# Plot original spectrum and fitted Gaussian
plt.figure(figsize=(10, 6))
plt.plot(wavenumbers, intensities, label='Original Spectrum')
plt.plot(wavenumbers, fit_curve, label='Gaussian Fit', color='red')

plt.xlabel('Wavenumber')
plt.ylabel('Intensity')
plt.title('FTIR Spectrum with Gaussian Fit')
plt.legend()
plt.grid(True)
plt.show()

# Print fitted parameters
print("Fitted parameters (Amplitude, Mean, Stddev):", popt)
