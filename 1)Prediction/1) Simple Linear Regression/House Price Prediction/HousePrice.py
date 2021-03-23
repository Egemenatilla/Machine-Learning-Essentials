#Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visiualization
from scipy.stats import pearsonr # correlation

#Import data from csv file
data = pd.read_csv('kc_house_data.csv')

#Correlation Between Features vs Price

# We try to find features vs price.
features = data.iloc[:,3:].columns.tolist()
print(features)
price = data.iloc[:,2].name
corr = {}
for i in features:
    temp = data[[i,price]]
    x1 = temp[i].values
    x2 = temp[price].values
    key = i + ' vs ' + price
    corr[key] = pearsonr(x1,x2)[0]
data_corr = pd.DataFrame(corr, index=['Value']).T
data_corr.loc[data_corr['Value'].abs().sort_values(ascending=False).index]

#Slicing
x = data[['sqft_living']]
y = data[['price']]

#Simple Linear Regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

#Visualization
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,lin_reg.predict(X_train),color = 'blue')
plt.xlabel('Price')
plt.ylabel('sqft_living',rotation = 0)
plt.title("sqft_living vs price (Train)")