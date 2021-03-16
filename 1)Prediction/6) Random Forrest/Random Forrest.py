# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:44:11 2021

@author: egeme
"""

#Kütüphane
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Veri Setini okuma
data = pd.read_csv('maaslar.txt')

#X ve Y eksenlerini ayırma
x = data[['Egitim Seviyesi']]
X = x.values

y = data[['maas']]
Y = y.values

#import RandomForrest in sklearn

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10,random_state=0) #estimators kaç farklı eleman varsa onu veriyorsun sayı olarak.

rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.5]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')



