
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

A=pd.read_csv("Sales.csv")

A.head()

A.tail()

A.info()

A.describe()

A.describe(include='all')

A.columns

print(A.shape)

print(A.nunique())

A.isna().sum()

len(A)

sb.pairplot(A)

sb.boxplot(A.COUNTRY,A.SALES)

sb.boxplot(A.PRICEEACH,A.SALES)

sb.boxplot(A.QUANTITYORDEREDNTRY,A.SALES)

sb.boxplot(A.MSRP,A.SALES)

from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
B = A.apply(le.fit_transform)
print(B)

Y = B[["SALES"]]
X = B[["PRICEEACH","QUANTITYORDERED","MSRP","PODUCTCODE", "CUSTOMERNAME","STATUS","DEALSIZE","COUNTRY","CITY"]]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=8)

sb.distplot(B.SALES)
sb.distplot(ytrain.SALES)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model1 = lm.fit(xtrain,ytrain)
pred = model1.predict(xtest)
xtest["predicted value"] = pred
xtest["Actual value"] = ytest

print(xtest)

import statsmodels.api as sm
X2 = sm.add_constant(xtrain)
est = sm.OLS(ytrain, X2)
est2 = est.fit()
print(est2.summary())

from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score

mean_squared_error(ytest,pred)

explained_variance_score(ytest,pred)

mean_absolute_error(ytest,pred)

r2_score(ytest,pred)

