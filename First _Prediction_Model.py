def Pred(age, sal):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Importing Dataset
    dataset = pd.read_csv('E:/Projects/Machine_Learning/Datasets/Social_Network_Ads.csv')
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting Dataset Into Training and Test set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    #print(x_train)

    # Training Model on Dataset
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train, y_train)

    # Predicting Results
    ans = classifier.predict(sc.transform([[age, sal]]))

    return ans

print("\nHere is a model to Predict whether a Customer of a particular age and salary\nis interested in buying A brand new SUV")

age = int(input('\nEnter the age : '))
sal = int(input('Enter the Salary : '))

pred = Pred(age, sal)

if pred==0:
    print("\n Prediction : The Customer does not seem Interested !")
else:
    print('\nPrediction : Hurray ! Congratulations on your new SUV')