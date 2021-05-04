# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Data from Data Set
data = pd.read_csv('Mall_Customers.csv')

# Drop Unnecessary Columns
X = data.drop(['CustomerID','Gender'],axis = 1)
print(X.head())

# Dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(X,method = "ward")
dendrogram(merg)
plt.xlabel("Data Point")
plt.ylabel("Euclidean Distance")
plt.show()

# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 5,affinity = "euclidean",linkage = "ward")
cluster = hc.fit_predict(X)

X["Label"] = cluster

sns.scatterplot(X['Annual Income (k$)'], X['Spending Score (1-100)'], hue=X['Label'])