# 출력을 원할 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

import pandas as pd

a = pd.read_csv("dataset/mtcars.csv")

# 사용자 코딩

print(a.head(3))

import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.model_selection import *
from scipy.stats import *

a['Unnamed: 0'] = LabelEncoder().fit_transform(a['Unnamed: 0'])

a = a.astype({'qsec':'int'})

a = a.dropna()
print(a.isna().sum())

x = a.drop('qsec',axis = 1)
y = a['qsec']

trainX, testX, trainY, testY = train_test_split(x,y,test_size = 0.25)

model = DecisionTreeClassifier(max_depth = 4)

model.fit(trainX,trainY)

pred = model.predict(testX)
# print(confusion_matrix(testY,pred,labels=[1,0]))
print("accuracy", accuracy_score(testY,pred))
print("recall", recall_score(testY,pred))
print("precision",precision_score(testY,pred))
print("Roc", roc_auc_score(testY,pred))
print("f1", f1_score(testY,pred))




# 해당 화면에서는 제출하지 않으며, 문제 풀이 후 답안제출에서 결괏값 제출