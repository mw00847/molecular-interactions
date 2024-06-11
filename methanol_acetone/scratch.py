import pandas as pd
from sklearn.cluster import KMeans
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

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3)  # You can adjust the number of clusters
df['Cluster'] = kmeans.fit_predict(df)

# Display the cluster assignments
print(df[['Cluster']])

# You can also plot the clusters if you have more than two dimensions
# (Note: For 2D or 3D visualization, you can choose any two or three columns from df)
plt.scatter(df[columns[0]], df[columns[1]], c=df['Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Spectral Data')
plt.xlabel(columns[0])
plt.ylabel(columns[1])
plt.show()
