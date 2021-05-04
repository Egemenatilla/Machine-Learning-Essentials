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

# Looking for Elbows
from sklearn.cluster import KMeans

wcss = []

for k in range(1,15):
    kmeans= KMeans(n_clusters = k)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15),wcss)
plt.xlabel("Number Of K Value")
plt.ylabel("WCSS")
plt.show()

# Kmeans Clustering
# We have two possible elbow (3 and 5)
# 3
kmeans3 = KMeans(n_clusters = 3)
clusters = kmeans3.fit_predict(X)
X["Label"] = clusters

sns.scatterplot(X['Annual Income (k$)'], X['Spending Score (1-100)'], hue=X['Label'],palette=sns.color_palette('hls', 3))
plt.xlabel('Income')
plt.ylabel('Score')
plt.title('3 Clusters')
plt.show()
#------------------------
# 5
kmeans3 = KMeans(n_clusters = 5)
clusters = kmeans3.fit_predict(X)
X["Label"] = clusters

sns.scatterplot(X['Annual Income (k$)'], X['Spending Score (1-100)'], hue=X['Label'],palette=sns.color_palette('hls', 5))
plt.xlabel('Income')
plt.ylabel('Score')
plt.title('5 Clusters')
plt.show()