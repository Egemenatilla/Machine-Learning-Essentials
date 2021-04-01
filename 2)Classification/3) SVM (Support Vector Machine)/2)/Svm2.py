# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 20:42:24 2021

@author: egeme
"""
# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data from csv file
data = pd.read_csv('data.csv')

# Drop the columns we cannot use
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

# scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="bad",alpha= 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="good",alpha= 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# Apply label encode to diagnosis column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

#Slicing
x_data = data.drop(["diagnosis"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
y = data.diagnosis.values

# Split train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

# SVM
from sklearn.svm import SVC 
svm = SVC(random_state = 1)
svm.fit(x_train,y_train)

# Accuracy
print("Accuracy of svm: ",svm.score(x_test,y_test))