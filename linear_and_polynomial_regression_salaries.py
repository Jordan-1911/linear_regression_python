import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importing the dataset/reading the csv file and showing first 5
df = pd.read_csv('position_salaries.csv')
print(df.head())

# Now, we can split the data into features and target
# X is the level column (presumably years of experience)
# y is the salary column

X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

# Now, we can split the dataset and get training set
# and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Now, we can fit the linear regression model to the dataset
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)


def linear_plot():
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, linear_regressor.predict(X_train), color='blue')
    plt.ticklabel_format(style='plain')
    plt.title('Salary VS Position Level (Training set)')
    plt.xlabel('Position Level')
    plt.ylabel('Salary')
    plt.show()
    return


linear_plot()

# We can see that linear regression does not fit the data well

y_pred = linear_regressor.predict([[5]])
print(y_pred)  # it predicts a $210,675 salary for level 5

# Now, we can fit the polynomial regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures

polynomial_regressor = PolynomialFeatures(degree=4)
X_polynomial = polynomial_regressor.fit_transform(X)
pol_regressor = LinearRegression()
pol_regressor.fit(X_polynomial, y)


def polynomial_plot():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_regressor.predict(polynomial_regressor.fit_transform(X)), color='blue')
    plt.ticklabel_format(style='plain')
    plt.title('Salary VS Position Level (Polynomial Regression)')
    plt.xlabel('Position Level')
    plt.ylabel('Salary')
    plt.show()
    return


polynomial_plot()

# Now, we can predict a new result with polynomial regression with more accuracy
y_poly_pred = pol_regressor.predict(polynomial_regressor.fit_transform([[5]]))
print(y_poly_pred)  # it predicts a $121,724 salary for level 5

print(" Using Linear Simple linear regression,"
      " the salary for a level 5 employee is: ", y_pred)
print(" Using Polynomial Regression,"
      " the salary for a level 5 employee is: ", y_poly_pred)

# As we can see, polynomial regression fits the data better than linear regression
