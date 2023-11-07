import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.model_selection import *
from matplotlib import pyplot as plt
from scipy.stats import iqr

train = pd.read_csv('dataset/Hitters.csv')
print(train.describe())
print(train.info())
print(train.head())
print(train.columns)

print(train.isna().sum())
train['Salary'] = train['Salary'].fillna(train['Salary'].mean())
# train = train.dropna()
print(train.isna().sum())

train = pd.get_dummies(train,drop_first=True)
print(train.head())
print(train.columns)

x = train.drop('Salary',axis= 1)
y = train['Salary']

trainX,testX,trainY,testY = train_test_split(x,y,test_size=0.2)
# 8:2 비율로 trian 8 test 2 할당됨

model = LinearRegression()
model.fit(trainX,trainY)

#독립변수의 순서를 출력
print(model.feature_names_in_)

#회귀분석계수출력
print(model.coef_)

#절편값
print(model.intercept_)

pred = model.predict(testX)

print(mean_squared_error(testY,pred,squared=False))