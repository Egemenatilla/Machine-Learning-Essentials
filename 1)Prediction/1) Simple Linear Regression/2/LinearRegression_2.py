#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the data from csv file
data = pd.read_csv('dataset.csv',sep = ';')

#Slicing
x = data.iloc[:,0:1]
y = data.iloc[:,1:]

#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Visualization
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')