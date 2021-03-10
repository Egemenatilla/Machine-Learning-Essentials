# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 20:48:59 2021

@author: egeme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read data from csv file and create a data frame
data = pd.read_csv('satislar.csv')
print(data)

#Separate the columns
months = data[['Aylar']]
sales = data[['Satislar']]
print(months)
print(sales)

#Separating data for training and testing.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(months,sales,test_size = 0.33,random_state = 0)

#Model Building and applying linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

lr.predict(x_test) 

#Visualization
#Sort the data
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.show()

plt.plot(x_test,lr.predict(x_test))

plt.title('Sales by month')
plt.xlabel('Months')
plt.ylabel('Sales')

