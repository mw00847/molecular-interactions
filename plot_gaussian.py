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

