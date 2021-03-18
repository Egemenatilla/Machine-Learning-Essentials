#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the data from csv file
data = pd.read_csv('decision_tree.csv',sep = ';')

#Slicing
x = data[['Seviye']] #data.iloc[:,0:1]
y = data[['Fiyat']] #data.iloc[:,1:]

#Import Decision Tree from sklearn
from sklearn.tree import DecisionTreeRegressor
dec_reg = DecisionTreeRegressor(random_state = 0)
dec_reg.fit(x,y)

#Visualization
plt.scatter(x,y,color = 'red')
plt.plot(x,dec_reg.predict(x),color = 'blue')
plt.xlabel('Level')
plt.ylabel('Price')

#print(dec_reg.predict([[5.5]]))