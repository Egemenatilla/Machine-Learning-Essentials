# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 21:47:26 2021

@author: egeme
"""

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read Data from csv file
data = pd.read_csv('veriler.csv')

# Slicing 
x = data.iloc[:,1:4]
y = data.iloc[:,4:]
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

# import GaussianNB from naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train.ravel())

y_pred = gnb.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

# Accuracy
print("Accuracy: %",accuracy_score(y_test, y_pred)*100)
