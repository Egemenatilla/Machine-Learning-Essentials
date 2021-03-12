# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:44:07 2021

@author: egeme
"""
#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the data from csv file
data = pd.read_csv('maaslar.txt')

#Slicing
#X axis
x = data[['Egitim Seviyesi']]
#print(x)
X = x.values
#Y axis
y = data[['maas']]
#print(y)
Y = y.values

#Linear Regression
from sklearn.linear_model import LinearRegression #20-26 lineer regresyon
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Polynomial Regression -Non Linear-
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) #degree = 2 means we use second degree polynomial.
#Our attitude here is this.Express it as a polynomial first, then apply regression.
x_poly = poly_reg.fit_transform(X)
#print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#Visualization
#Linear Regression
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.show()
#Polynomial Regression
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.show()
