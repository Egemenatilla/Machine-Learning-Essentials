# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:07:14 2021

@author: egeme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('tennis.csv')
print(data)

#We have 3 non numeric columns. We apply OneHotEncode on the outlook. 
#We apply LabelEncode on windy and play. Because there are only two property.

from sklearn.preprocessing import LabelEncoder

data2 = data.apply(LabelEncoder().fit_transform) # This line apply LabelEncode all columns.

#We do not want to apply Label encode numeric columns o we will take care this later.
#OneHotEncode  outlook column
o = data2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories='auto')
o = ohe.fit_transform(o).toarray()
print(o)

weather_forecast = pd.DataFrame(data = o, index = range(14),columns = ['outlook','rainy','sunny'])
lastData = pd.concat([weather_forecast,data.iloc[:,1:3]],axis = 1)
lastData = pd.concat([data2.iloc[:,-2:],lastData],axis = 1)

# Separating data for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(lastData.iloc[:,:-1],lastData.iloc[:,-1:],test_size=0.33, random_state=0)
#Applyin Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)

#Improvement with using backward elimination
import statsmodels.api  as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=lastData.iloc[:,:-1], axis=1 )
X_l = lastData.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = lastData.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

#Removed 0. column
lastData = lastData.iloc[:,1:]

import statsmodels.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=lastData.iloc[:,:-1], axis=1 )
X_l = lastData.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = lastData.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)