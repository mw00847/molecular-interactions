import numpy as np
import matplotlib.pyplot as plt

"""
x is the variable
μ is the mean of the distribution. It determines the center of the Gaussian curve on the x-axis.
σ is the standard deviation. It controls the spread or width of the curve. A larger standard deviation means a broader curve, while a smaller standard deviation means a narrower curve.
A is the amplitude or peak height of the curve. It scales the curve
"""

# Define the Gaussian function
def gaussian(x, mu, sigma, A):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Generate some sample data
np.random.seed(0)  # for reproducibility
x_data = np.linspace(-5, 5, 100)
y_data = gaussian(x_data, 0, 1, 1) + np.random.normal(0, 0.1, size=x_data.shape)

# Plot the sample data
plt.scatter(x_data, y_data)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sample Data for Gaussian Curve')
plt.show()



# Fit the Gaussian curve to the data using least squares optimization
def gaussian_residuals(params, x, y):
    mu, sigma, A = params
    return y - gaussian(x, mu, sigma, A)

from scipy.optimize import least_squares

initial_guess = [0, 1, 1]  # initial guess for mu, sigma, A
result = least_squares(gaussian_residuals, initial_guess, args=(x_data, y_data))

# Extract the optimized parameters
mu_opt, sigma_opt, A_opt = result.x

# Generate the fitted Gaussian curve
y_fit = gaussian(x_data, mu_opt, sigma_opt, A_opt)

# Plot the original data and the fitted curve
plt.scatter(x_data, y_data, label='Original Data')
plt.plot(x_data, y_fit, color='red', label='Fitted Gaussian Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitted Gaussian Curve')
plt.legend()
plt.show()

# Print the optimized parameters
print(f'Optimized Parameters:')
print(f'Mean (μ): {mu_opt:.2f}')
print(f'Standard Deviation (σ): {sigma_opt:.2f}')
print(f'Amplitude (A): {A_opt:.2f}')