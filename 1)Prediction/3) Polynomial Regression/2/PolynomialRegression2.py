#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read data from csv file
data = pd.read_csv('polynomial_regression.csv',sep = ';')

#Slicing
x = data.iloc[:,0:1]
y = data.iloc[:,1:]

#Import linear and polynomial regressions
from sklearn.linear_model import LinearRegression #20-26 lineer regresyon
lin_reg = LinearRegression()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x.values)
lin_reg.fit(x_poly,y)

#Visualization
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(poly_reg.fit_transform(x)),color='blue')
plt.show()

