#this takes the FTIR of mixtures of ethylene gylcol, glycerine and water and uses PCA 

import pandas as pd 
from sklearn.decomposition import PCA
import numpy
import matplotlib.pyplot as plt

#df1 array is the wavenumbers

df1=pd.read_csv('EG_25.csv',usecols=[0])

#these are the FTIR of the mixtures 
df2=pd.read_csv('EG_25.csv',usecols=[1])
df3=pd.read_csv('EG_5050.csv',usecols=[1])
df4=pd.read_csv('EG_75.csv',usecols=[1])
df5=pd.read_csv('G_100.csv',usecols=[1])
df6=pd.read_csv('G_25.csv',usecols=[1])
df7=pd.read_csv('G_50.csv',usecols=[1])
df8=pd.read_csv('G_75.csv',usecols=[1])
df10=pd.read_csv('water.csv',usecols=[1])


#concat the dataframes into one and name the columns
df=pd.concat([df2,df3,df4,df5,df6,df7,df8,df10,df11,df12],axis=1)
df.columns=["EG25","EG5050","EG75","G100","G25","G50","G75","water"]


#normalise the data and running PCA
df_normalized=(df - df.mean()) / df.std()
pca = PCA(n_components=df.shape[1])
pca.fit(df_normalized)

loadings = pd.DataFrame(pca.components_.T,
columns=['PC%s' % _ for _ in range(len(df_normalized.columns))],
index=df.columns)

#plot the variance explained by PCA 
plot.plot(pca.explained_variance_ratio_)
plot.ylabel('Explained Variance')
plot.xlabel('Components')

plot.show()

#plotting the scatter plot
plt.scatter(loadings.loc[:,"PC1"],loadings.loc[:,"PC2"])




plt.scatter(loadings.loc["EG_25","PC1"],loadings.loc["EG_25","PC2"],marker="*")
plt.scatter(loadings.loc["EG5050","PC1"],loadings.loc["EG5050","PC2"],marker="o")
plt.scatter(loadings.loc["EG75","PC1"],loadings.loc["EG75","PC2"],marker="1")
plt.scatter(loadings.loc["G100","PC1"],loadings.loc["G100","PC2"],marker=">")
plt.scatter(loadings.loc["G25","PC1"],loadings.loc["G25","PC2"],marker="2")
plt.scatter(loadings.loc["G50","PC1"],loadings.loc["G50","PC2"],marker="3")
plt.scatter(loadings.loc["G75","PC1"],loadings.loc["G75","PC2"],marker="8")
plt.scatter(loadings.loc["water","PC1"],loadings.loc["water","PC2"],marker="x")

plt.legend(["EG25","EG5050","EG75","G100","G25","G50","G75","water"])

plt.xlabel("PC1")
plt.ylabel("PC2")
