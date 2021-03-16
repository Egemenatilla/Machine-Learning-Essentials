# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 21:51:51 2021

@author: egeme
"""

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the data from csv file
data = pd.read_csv('multipleLinearRegression.csv',sep = ';')

#Slicing
x = data.iloc[:,:1]
y = data.iloc[:,2:] #Before Backward Elimination y = data.iloc[:,1:]
X = x.values
Y = y.values

#Import Linear Regression
from sklearn.linear_model import LinearRegression
multiple_lin_reg = LinearRegression()
multiple_lin_reg.fit(x,y)

#First Backward Elimination
# import statsmodels.api as sm 
# X = np.append(arr = np.ones((14,1)).astype(int), values=data.iloc[:,:-1], axis=1 )
# X_l = data.iloc[:,[0,1,2]].values
# r_ols = sm.OLS(endog = data.iloc[:,-1:], exog =X_l)
# r = r_ols.fit()
# print(r.summary())

#Second Backward Elimination
import statsmodels.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=data.iloc[:,:-1], axis=1 )
X_l = data.iloc[:,[0,2]].values
r_ols = sm.OLS(endog = data.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())
#Note: We looked the P value. Then eliminated some features from data set.(Salary column's P value = 0.915)

#Visualization
plt.scatter(x,y,color = 'red')
plt.plot(x,multiple_lin_reg.predict(x),color = 'blue')
