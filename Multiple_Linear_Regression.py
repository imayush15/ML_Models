import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv('E:\Machine_Learning\Datasets\Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encodinf Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = ct.fit_transform(x)

# Splitting Dataset into Training and Test Dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression Model on Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting Results of Test set
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)  # it is used to confine the number of digits after decimal places to 2
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))
# As y_pred is 1-D horizontal array, reshape is used to change it to vertical array with number of rows len(y_pred)
# and 1 column also line 29 is used to compare the y_pred values to that of y_test values
