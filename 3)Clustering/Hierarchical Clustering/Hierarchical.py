# Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

# Create Dataset
# class1
x1 = np.random.normal(25,5,100)
y1 = np.random.normal(25,5,100)
# class2
x2 = np.random.normal(55,5,100)
y2 = np.random.normal(60,5,100)
# class3
x3 = np.random.normal(55,5,100)
y3 = np.random.normal(15,5,100)
# concatenate
x = np.concatenate((x1,x2,x3),axis = 0)
y = np.concatenate((y1,y2,y3),axis = 0)

dictionary = {"x":x,"y":y}
# Dataset
data = pd.DataFrame(dictionary)

# Dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data,method = "ward")
dendrogram(merg)
plt.xlabel("Data Point")
plt.ylabel("Euclidean Distance")
plt.show()

# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 3,affinity = "euclidean",linkage = "ward")
cluster = hc.fit_predict(data)

data["Label"] = cluster
# Visualization
plt.scatter(data.x[data.Label == 0],data.y[data.Label == 0],color = "red")
plt.scatter(data.x[data.Label == 1],data.y[data.Label == 1],color = "blue")
plt.scatter(data.x[data.Label == 2],data.y[data.Label == 2],color = "green")
plt.show()

