import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.ensemble import *
from sklearn.model_selection import *
from sklearn.tree import *

train = pd.read_csv('dataset/PimaIndiansDiabetes2.csv')

print(train.columns)

print(train.isna().sum())

print(train.info())

train['diabetes'] = LabelEncoder().fit_transform(train['diabetes'])

train = train.fillna(train.mean())

x = train.drop('diabetes',axis=1)
y = train['diabetes']

base = DecisionTreeClassifier(max_depth=4)
model = BaggingClassifier(base, n_estimators=500)

trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2)

model.fit(trainX,trainY)

pred = model.predict(testX)

print(recall_score(testY,pred))
print(accuracy_score(testY,pred))
print(precision_score(testY,pred))
print(f1_score(testY,pred))
print(roc_auc_score(testY,pred))