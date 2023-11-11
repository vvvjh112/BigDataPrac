import pandas as pd
import numpy as np
from scipy.stats import iqr
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.preprocessing import *

#종속변수가 범주형인 경우 로지스틱 회귀 분석
train = pd.read_csv('dataset/PimaIndiansDiabetes2.csv')
print(train.info())
print(train.isna().sum())

train['diabetes'] = LabelEncoder().fit_transform(train['diabetes'])

#결측치 평균 대체 혹은 삭제
# train = train.fillna(train.mean())
train = train.dropna()

x = train.drop('diabetes',axis=1)
y = train['diabetes']

trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2)

model = LogisticRegression(max_iter=1000)

# md = LogisticRegression()
# params = {'max_iter' : [200,400,600,800,1000,1200]}
# gv = GridSearchCV(md,params, n_jobs = -1, cv = 4)
# gv.fit(trainX,trainY)
# print("GV",gv.best_params_)

model.fit(trainX,trainY)

pred = model.predict(testX)



print(confusion_matrix(testY,pred,labels=[1,0]))
print("Recall",recall_score(testY,pred))
print("Accuracy",accuracy_score(testY,pred))
print("Precision",precision_score(testY,pred))
print("F1Score",f1_score(testY,pred))
print("Roc Auc Score",roc_auc_score(testY,pred))