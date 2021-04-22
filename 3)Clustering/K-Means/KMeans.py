# Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

# Create Dataset
# class1
x1 = np.random.normal(25,5,1000)
y1 = np.random.normal(25,5,1000)
# class2
x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)
# class3
x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)
# concatenate
x = np.concatenate((x1,x2,x3),axis = 0)
y = np.concatenate((y1,y2,y3),axis = 0)

dictionary = {"x":x,"y":y}
# Dataset
data = pd.DataFrame(dictionary)

# Visualization
plt.scatter(x1,y1,color = "black")
plt.scatter(x2,y2,color = "black")
plt.scatter(x3,y3,color = "black")
plt.show()

# KMeans
from sklearn.cluster import KMeans
wcss = []

for k in range(1,15):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,15),wcss)
plt.xlabel("Number Of K Value")
plt.ylabel("WCSS")
plt.show()

# After last plot we decided k = 3

kmeans2 = KMeans(n_clusters = 3)
clusters = kmeans2.fit_predict(data)
data["Label"] = clusters

plt.scatter(data.x[data.Label == 0],data.y[data.Label == 0],color = "red")
plt.scatter(data.x[data.Label == 1],data.y[data.Label == 1],color = "blue")
plt.scatter(data.x[data.Label == 2],data.y[data.Label == 2],color = "green")
plt.show()



