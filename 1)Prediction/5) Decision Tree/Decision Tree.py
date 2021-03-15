# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:34:57 2021

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
X = x.values
y = data[['maas']]
Y = y.values

#Import Decision Tree
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

#Visualization
plt.scatter(X, Y, color='red')
plt.plot(X,r_dt.predict(X),color='blue')