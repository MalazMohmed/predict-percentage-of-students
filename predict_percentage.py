# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:58:49 2022

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset=pd.read_csv('student_scores.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values

#spliting dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=1/3, random_state=0)


#fitting simple linear regression on training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set result
y_pred=regressor.predict(x_test)

# evaluate the model
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 

#visualising traning set result
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary Vs Experience (Training set￼)')
plt.xlabel("year of experience")
plt.ylabel("Salary")
plt.show()

#visualising test set result
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary Vs Experience (Test set)')
plt.xlabel("year of experience")
plt.ylabel("Salary")
plt.show()

