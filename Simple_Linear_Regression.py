import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing Dataset
dataset = pd.read_csv('E:/Machine_Learning/Datasets/Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting into test and training set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Using Linear Regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting Results of Test set
y_pred = regressor.predict(x_test)

# Plotting Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Result Of Training Set')
plt.xlabel('Years of Experience')
plt. ylabel('Salary')
plt.show()

# Plotting Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, y_pred, color='blue')
plt.title('Result Of Test Set')
plt.xlabel('Years of Experience')
plt. ylabel('Salary')
plt.show()