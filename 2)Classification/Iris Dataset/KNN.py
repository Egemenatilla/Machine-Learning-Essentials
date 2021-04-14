# Libraries
import numpy as np # linear algebra
import pandas as pd 


# Read data from csv
data = pd. read_csv('Iris.csv')

# Slicing
x = data.iloc[:,1:5]
y = data.iloc[:,5:]

# Label Encoding for "Species" column

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y)

# Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)

# Fitting the model
knn.fit(X_train, y_train)

# Predicting the Test set results
y_pred = knn.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cm

# Accuracy
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy: %',accuracy)

