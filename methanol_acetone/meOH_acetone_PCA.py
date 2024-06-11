import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Read the data
df1 = pd.read_csv('a.csv', usecols=[0])
df3 = pd.read_csv('b.csv', usecols=[1])
df4 = pd.read_csv('c.csv', usecols=[1])
df6 = pd.read_csv('e.csv', usecols=[1])
df7 = pd.read_csv('f.csv', usecols=[1])
df8 = pd.read_csv('g.csv', usecols=[1])
df9 = pd.read_csv('h.csv', usecols=[1])
df10 = pd.read_csv('i.csv', usecols=[1])
df11 = pd.read_csv('j.csv', usecols=[1])

# Concatenate dataframes
df = pd.concat([df3, df4, df6, df7, df8, df9, df10, df11], axis=1)
columns = ["b", "c", "e", "f", "g", "h", "i", "j"]

# Apply PCA
pca = PCA(n_components=df.shape[1])
X_pca = pca.fit_transform(df)

# Plot the original data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for col in df.columns:
    plt.plot(np.array(df1), df[col], 'o', label=f'Sample {col}')

plt.title('Original Data')
plt.xlabel('Wavenumber')
plt.ylabel('Absorbance')
plt.legend()

# Plot the PCA-transformed data
plt.subplot(1, 2, 2)
for i in range(df.shape[1]):
    plt.plot(X_pca[:, i], 'o', label=f'PC{i + 1}')

plt.title('PCA Transformed Data')
plt.xlabel('Principal Components')
plt.ylabel('Transformed Values')
plt.legend()

plt.tight_layout()
plt.show()


