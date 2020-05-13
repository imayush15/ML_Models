import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset

dataset = pd.read_csv('E:\Machine_Learning\Datasets\Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the linear regression model on the whole dataset
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Training the Polynomial Regression Model On the whole dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#Visualizing Linear Regression Results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Linear Regression Result')
plt.xlabel('Position ------->')
plt.ylabel('Salary --------->')
#plt.show()

#Visualization of Polynomial Regression Results
plt.scatter(x, y, color='blue')
plt.plot(x, lin_reg_2.predict(x_poly), color='red')
plt.title('Polynomial Linear Regression Result')
plt.xlabel('Position ------->')
plt.ylabel('Salary --------->')
#plt.show()

#Predicting a result with Linear Regression
res = lin_reg.predict([[6.5]])
#print(res)

#Predicting a result with Polynomial Linear Regression
res2 = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(res2)