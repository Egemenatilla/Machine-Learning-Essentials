# Libraries
import numpy as np 
import pandas as pd

# Read data from csv file
data = pd. read_csv('Iris.csv')

# Slicing
x = data.iloc[:,1:5]
y = data.iloc[:,5:]

# Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

# Label Encoding for "Species" column
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y.ravel())

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(X_train,y_train)

# Accuracy
y_pred = logr.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test)
