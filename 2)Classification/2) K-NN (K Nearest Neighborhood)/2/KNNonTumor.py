# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Read data 
data = pd.read_csv('data.csv')

# Drop unnecessary columns
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

# Apply label encode to diagnosis column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

#Slicing
x_data = data.drop(["diagnosis"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
y = data.diagnosis.values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train)

# Accuracy
prediction = knn.predict(x_test)
print("Accuracy : ",knn.score(x_test,y_test))

# Visualization
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="bad",alpha= 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="good",alpha= 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
