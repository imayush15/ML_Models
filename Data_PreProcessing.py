import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
data = pd.read_csv('Data.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


#Taking Care of Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #Replacing the missing values by mean of other values
imputer.fit(x[:, 1:3]) #fit method finds out all the missing values in the specifies rows and columns
x[:, 1:3] =imputer.transform(x[:, 1:3]) #transform method is used to transform the rows and columns

#print(x)

#Encoding The Categorical Data
#-------->Encoding Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
#Object for ColumnTransformer, which takes argument what to transform and what to leave

x = ct.fit_transform(x)
#print(x)

#---------->Encoding Dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#print(y)

#Splitting Dataset Into Training and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#print(x_train, x_test, y_train, y_test)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train[:, 3:])
x_train[:,3:] = sc.transform(x_train[:,3:])
print(x_train)
