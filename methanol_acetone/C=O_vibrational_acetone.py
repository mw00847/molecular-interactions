import pandas as pd
from sklearn.decomposition import PCA
import numpy
import matplotlib.pyplot as plt


from sklearn import decomposition

#df1 is the wavenumbers
df1=pd.read_csv('a.csv',usecols=[0])

#these are the variables
df2=pd.read_csv('a.csv',usecols=[1])
df3=pd.read_csv('b.csv',usecols=[1])
df4=pd.read_csv('c.csv',usecols=[1])
#df5=pd.read_csv('d.csv',usecols=[1])
df6=pd.read_csv('e.csv',usecols=[1])
df7=pd.read_csv('f.csv',usecols=[1])
df8=pd.read_csv('g.csv',usecols=[1])
df9=pd.read_csv('h.csv',usecols=[1])
df10=pd.read_csv('i.csv',usecols=[1])
df11=pd.read_csv('j.csv',usecols=[1])
df12=pd.read_csv('k.csv',usecols=[1])

#concat the dataframes into one and name the columns

df=pd.concat([df2,df3,df4,df6,df7,df8,df9,df10,df11,df12],axis=1)

columns=(["a","b","c","e","f","g","h","i","j","k"])

print(df)