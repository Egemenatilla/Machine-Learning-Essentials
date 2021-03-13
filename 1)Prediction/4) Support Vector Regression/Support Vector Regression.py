# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 18:44:16 2021

@author: egeme
"""
#Libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#Read the data from csv file
data = pd.read_csv('maaslar.txt')

#Slicing
x = data[['Egitim Seviyesi']]
y = data[['maas']]

#Scaling
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_scale = sc1.fit_transform(x)
sc2 = StandardScaler()
y_scale = np.ravel(sc2.fit_transform(y.values.reshape(-1,1)))

#Applying SVR to model.
from sklearn.svm import SVR #Import SVR(Support Vector Regression) from scikit learn

svr_reg = SVR(kernel='rbf') #rbf mean is Radial basis function. This function uses here.
svr_reg.fit(x_scale,y_scale)

#Visualization
plt.scatter(x_scale,y_scale,color = 'red')
plt.plot(x_scale,svr_reg.predict(x_scale),color = "blue")


