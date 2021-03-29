# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('veriler.csv')

# Slicing
x = data.iloc[:,1:4] #independent variable
y = data.iloc[:,4:] #dependent variable

X = x.values
Y = y.values

# Dividing data for training and testing
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.33,random_state = 0)

# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state = 0)
logr.fit(X_train,y_train.ravel())

# Prediction
y_pred = logr.predict(X_test)
print(y_pred)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)