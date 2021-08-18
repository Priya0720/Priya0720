import matplotlib.pyplot as plt
import scipy
from scipy import stats
import numpy as np
import csv
from pandas import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm


def r_squared_value_linear_regression(cargo):
    x=cargo['OI']
    x=x.to_numpy()
    x=x.reshape(-1,1)
    y=cargo['VWAP']
    z=cargo['COMPANY']
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    b1 = model.coef_
    b0 = model.intercept_
    y_pred = model.predict(x)
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.xlabel("OI")
    plt.ylabel("VWAP")
    plt.show()
    return r_sq, b1, b0


def r_squared_value_polynomial_regression(cargo):
    x = cargo['OI']
    x = x.to_numpy()
    x = x.reshape(-1, 1)
    y = cargo['VWAP']
    z = cargo['COMPANY']
    x_ = PolynomialFeatures(degree=2).fit_transform(x)
    model = LinearRegression().fit(x_, y)
    r_sq = model.score(x_, y)
    b2 = model.intercept_
    b1, b0 = model.coef_[0], model.coef_[1]
    y_pred = LinearRegression().fit(x_, y).predict(x_)
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.xlabel("OI")
    plt.ylabel("VWAP")
    plt.show()
    return r_sq, b2, b1, b0


def r_squared_value_statsapi(cargo):
    x = cargo['OI']
    x = x.to_numpy()
    x = x.reshape(-1, 1)
    y = cargo['VWAP']
    z = cargo['COMPANY']
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    print (results.summary())
    print ("\n")


# Read the data from a CSV File in a pandas DataFrame
oi_and_vwap = pd.read_csv("oi_combined.csv", header=None, usecols=[1, 2, 3, 4])

# Assigning legends to the column
oi_and_vwap.columns = ['OI', 'DATE', 'COMPANY', 'VWAP']

# Selecting by company name
list_of_companies = (oi_and_vwap['COMPANY'].unique())

while True :
    choice = int(input("Enter 1 for linear regression, 2 for polynomial regression, 3 for OLS summary: "))
    if choice == 1 :
        for value in list_of_companies:
            cargo = globals()[f"data_for_{value}"] = oi_and_vwap.loc[oi_and_vwap['COMPANY'] == value]
            r_sq, b1, b0 = r_squared_value_linear_regression(cargo)
            print(value, "\n R Squared:", r_sq, "\n Relation: OI =", b0, "+", b1, "x VWAP")
    elif choice == 2 :
        for value in list_of_companies:
            cargo = globals()[f"data_for_{value}"] = oi_and_vwap.loc[oi_and_vwap['COMPANY'] == value]
            r_sq, b2, b1, b0 = r_squared_value_polynomial_regression(cargo)
            print(value, "\n R Squared:", r_sq, "\n Relation: OI =", b0, "+", b1, "(VWAP) +", b2, "(VWAP)^2")
    elif choice == 3 :
        for value in list_of_companies:
            cargo = globals()[f"data_for_{value}"] = oi_and_vwap.loc[oi_and_vwap['COMPANY'] == value]
            print (value)
            r_squared_value_statsapi(cargo)
    else :
        print ("Invalid choice")
    cont = input("do you wish to continue?(y/n):")
    if cont in "Nn" :
        break

