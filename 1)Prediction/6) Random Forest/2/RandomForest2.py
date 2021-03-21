#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read data from csv file
data = pd.read_csv('random_forest.csv',sep = ';')

#Slicing
x = data.iloc[:,:1]
y = data.iloc[:,1:2]

X = x.values
Y = y.values

#Import Random Forest Regressor from sklearn
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())

#Visualization
plt.scatter(X,Y,color = 'red')
plt.plot(x,rf_reg.predict(X),color = 'blue')