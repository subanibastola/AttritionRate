
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

df=pd.read_excel("Attrition.xlsx",sheet_name="HR-Employee-Attrition")

pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',500)

import os
print(os.listdir("\\Users\ADMIN\Documents\Data Science"))

matplotlib.style.use('ggplot')

df.head()

df.tail()

df.describe(include='all')

df.describe()

df.columns

print(df.shape)
print("*****************")

print(df.nunique())

print(df[df['Attrition'] == 1]["Attrition"].count())
print(df[df['Attrition'] == 0]["Attrition"].count())

len(df.columns)

df_one=df[df['Attrition'] == 1]["Age"]
df_zero=df[df['Attrition'] == 0]["Age"]

df.isna().sum()

df.drop_duplicates(keep='first')

len(df)

df=df.drop_duplicates(keep='first')

df['Gender'].value_counts().plot.bar(title="Gender")

df['BusinessTravel'].value_counts().plot.bar(title="Freq dist of BusinessTravel")

df['Department'].value_counts().plot.bar(title="Freq dist of Department")

df['MaritalStatus'].value_counts().plot.bar(title="MaritalStatus")

df['EducationField'].value_counts().plot.bar(title="Freq dist of EducationField") 

col_names = ["Age","DailyRate","DistanceFromHome","Education","EnvironmentSatisfaction","HourlyRate","JobInvolvement","JobLevel","JobSatisfaction","MonthlyIncome","MonthlyRate","NumCompaniesWorked","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"
            ]

fig, ax = plt.subplots(len(col_names), figsize=(16,12))

for i, col_val in enumerate(col_names):
        
    sns.distplot(df[col_val], hist=True, ax=ax[i])
    ax[i].set_title('Freq dist '+col_val, fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)
    ax[i].set_ylabel('Count', fontsize=8)
    
plt.show()

col_names = ["Age","DailyRate","DistanceFromHome","Education","EnvironmentSatisfaction","HourlyRate","JobInvolvement","JobLevel","JobSatisfaction","MonthlyIncome","MonthlyRate","NumCompaniesWorked","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"
]
fig, ax = plt.subplots(len(col_names), figsize=(8,40))

for i, col_val in enumerate(col_names):
        
    sns.boxplot(y=df[col_val], ax=ax[i])
    ax[i].set_title('Box plot - '+col_val, fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)
    
plt.show()


sns.pairplot(df)

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

df.dtypes

df=df.select_dtypes(include=['int64'])
df

len(df.columns)

df.head()

sns.boxplot(y=df['YearsSinceLastPromotion'],x=df['Attrition'])

sns.boxplot(y=df['Age'],x=df['Attrition'])

sns.boxplot(y=df['Education'],x=df['Attrition'])

sns.boxplot(y=df['YearsWithCurrManager'],x=df['Attrition'])

sns.boxplot(y=df['EnvironmentSatisfaction'],x=df['Attrition'])

sns.boxplot(y=df['Gender'],x=df['Attrition'])

sns.boxplot(y=df['JobInvolvement'],x=df['Attrition'])

def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval)

col_names=["MonthlyIncome","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager","PerformanceRating","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear"]
            

fig, ax = plt.subplots(len(col_names), figsize=(8,40))

for i, col_val in enumerate(col_names):
    x = df[col_val][:1000]
    sns.distplot(x, ax=ax[i], rug=True, hist=False)
    outliers = x[percentile_based_outlier(x)]
    ax[i].plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)

    ax[i].set_title('Outlier detection - '+col_val, fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)
    
plt.show()


df.info()

# Columns to remove 
remove_col_val = ["EmployeeCount","EmployeeNumber","StandardHours"]

y = df['Attrition']

df= df.drop(remove_col_val,axis=1)

# Converting type to categorical variable 

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df = df.apply(le.fit_transform)
df.info()

X = df[["Age","DailyRate","DistanceFromHome","HourlyRate","MonthlyIncome","MonthlyRate","NumCompaniesWorked","YearsInCurrentRole","TotalWorkingYears"]]
#X=df[df.columns[~df.columns.is["Attrition"])]]
Y = df[["Attrition"]]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,random_state = 50)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=50,criterion='entropy',max_leaf_nodes=73)
model_rfc=rfc.fit(x_train, y_train)
pred_rfc=model_rfc.predict(x_test)

from sklearn.metrics import accuracy_score
print('Validation Results')
as_rfc=accuracy_score(y_test,pred_rfc)
print(as_rfc)


from sklearn.model_selection import GridSearchCV
parameters = {"max_leaf_nodes": range(2,100,1)}


model = GridSearchCV(model_rfc,parameters,scoring="accuracy",cv=4)
model.fit(x_train,y_train)
model.best_params_


from sklearn.metrics import confusion_matrix, classification_report
cr=classification_report(pred_rfc,y_test)
cm=confusion_matrix(pred_rfc,y_test)
print(cr)
print(cm)

print(y_test[y_test['Attrition'] == 1]["Attrition"].count())

print(y_test[y_test['Attrition'] == 0]["Attrition"].count())

feature_importances = pd.DataFrame(clf_rf.feature_importances_,index = df.columns,columns=['importance']).sort_values('importance',ascending=False)
fig=plt.figure(figsize=(19,16))
import matplotlib.pyplot as p
p.barh(x_train.columns,model_rfc.feature_importances_)

