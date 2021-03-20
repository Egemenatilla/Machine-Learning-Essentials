# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 15:01:58 2021

@author: egeme
"""
#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read data from csv file
data = pd.read_csv('maaslar.txt')

x = data[['Egitim Seviyesi']]
X = x.values

y = data[['maas']]
Y = y.values

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

from sklearn.metrics import r2_score
print('Random Forest R2 value')
print(r2_score(Y,rf_reg.predict(X)))
print('Decision Tree R2 Value')
print(r2_score(Y,r_dt.predict(X)))
      