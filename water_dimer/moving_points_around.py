import numpy as np
import matplotlib.pyplot as plt

# Enable interactive plots if using in Jupyter, comment out if running as a script
# %matplotlib notebook

# Define points in 3D space
a = np.array([0.1, 0.2, 0.3])
b = np.array([0.18, 0.29, 0.35])
c = np.array([0.16, 0.27, 0.33])

# Plot initial points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(a[0], a[1], a[2], c='r', marker='o', label='Point A')
ax.scatter(b[0], b[1], b[2], c='g', marker='o', label='Point B')
ax.scatter(c[0], c[1], c[2], c='b', marker='o', label='Point C')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend(loc='upper left')
plt.title('Initial Points')
plt.xlim(0, 0.5)
plt.ylim(0, 0.5)
ax.set_zlim(0, 0.5)
plt.show()

# Rotate point a by 180 degrees around the x-axis
theta = np.pi  # 180 degrees in radians
rotation_matrix = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])
rotated_a = np.dot(rotation_matrix, a)



# Plot the points after rotation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(a[0], a[1], a[2], c='r', marker='*', label='Original Point A')
ax.scatter(rotated_a[0], rotated_a[1], rotated_a[2], c='orange', marker='o', label='Rotated Point A')
ax.scatter(b[0], b[1], b[2], c='g', marker='o', label='Point B')
ax.scatter(c[0], c[1], c[2], c='b', marker='o', label='Point C')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend(loc='upper left')
plt.title('Points After Rotating A')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define points
a = np.array([0.1, 0.2, 0.3])  # Point A
b = np.array([0.18, 0.29, 0.35])  # Point B
c = np.array([0.16, 0.27, 0.33])  # Point C

# Setup the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Loop through 360 degrees, rotating point B around the Y-axis 20 degrees at a time
for i in range(0, 360, 20):
    theta = np.radians(i)
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    rotated_b = np.dot(rotation_matrix, b)

    # Plot the rotated point
    ax.scatter(rotated_b[0], rotated_b[1], rotated_b[2], c='orange', marker='o',
               label=f'Rotated B at {i}°' if i == 0 else "")

# Plot constant points (A and C)
ax.scatter(a[0], a[1], a[2], c='r', marker='*', label='Point A')
ax.scatter(c[0], c[1], c[2], c='b', marker='o', label='Point C')

# Setting labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(loc='upper left')
ax.set_title('Rotations of Point B around the Y-axis')

# Show the plot
plt.show()




