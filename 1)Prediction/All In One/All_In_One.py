# Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import operator

# Read Data From csv File
data = pd.read_csv('bottle.csv')

# Slicing
new_data = data[['T_degC','Salnty']]
new_data = new_data[:][:1000]

# Drop NaN or missing input numbers
new_data.isnull().sum()
new_data.fillna(method='ffill', inplace=True)

x = new_data[['Salnty']]
y = new_data[['T_degC']]

# Split Data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

# Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()  
lin_reg.fit(X_train, y_train)

from sklearn.metrics import r2_score
y_pred =  lin_reg.predict(X_test)
print("Linear Regression R2 Score: ",r2_score(y_test,y_pred));

plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_test,y_pred,color = 'blue')
plt.xlabel("Temperature")
plt.ylabel("Salinity")
plt.title("Simple Linear Regression Model")
plt.show()

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y_train)

poly_pred = lin_reg2.predict(X_poly)
print("Polynomial Regression R2 Score :",r2_score(y_train,poly_pred))

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,poly_pred,color = 'blue')
plt.xlabel("Temperature")
plt.ylabel("Salinity")
plt.title("Polynomail Regression Model")
plt.show()

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train,y_train)

dt_pred = dt_reg.predict(X_train)

plt.scatter(X_train,y_train,color = 'red')
X_grid = np.arange(min(np.array(X_train)),max(np.array(X_train)), 0.01)  
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid,dt_reg.predict(X_grid),color="blue") 
plt.xlabel("Temperature")
plt.ylabel("Salinity")
plt.title("Decision Tree Model")
plt.show()

print("Decision Tree R2 Score: ",r2_score(y_train,dt_pred))

# Random Forest 
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X_train,y_train)

rf_pred = rf_reg.predict(X_train)

plt.scatter(X_train,y_train,color = 'red')
X_grid = np.arange(min(np.array(X_train)),max(np.array(X_train)), 0.01)  
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid,rf_reg.predict(X_grid),color = 'blue')
plt.xlabel("Temperature")
plt.ylabel("Salinity")
plt.title("Random Forest")
plt.show()

print("Random Forest R2 Score: ",r2_score(y_train,rf_pred))


































