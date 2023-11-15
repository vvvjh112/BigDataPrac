# 2 - 1

import pandas as pd
import numpy as np

train = pd.read_csv('dataset/P210204-01.csv')
test = pd.read_csv('dataset/P210204-02.csv')
tmp = pd.read_csv('dataset/P210204-02.csv')

#정시 도착여부 예측한 확률

print(train.info())
print(train.columns)
#['ID', 'Warehouse_block', 'Mode_of_Shipment', 'Customer_care_calls',
# 'Customer_rating', 'Cost_of_the_Product', 'Prior_purchases', 'Product_importance',
# 'Gender', 'Discount_offered', 'Weight_in_gms', 'Reached.on.Time_Y.N']

#라벨인코딩
from sklearn.preprocessing import *

encoding_ary = ['Warehouse_block','Mode_of_Shipment', 'Product_importance','Gender']

for i in encoding_ary:
    train[i] = LabelEncoder().fit_transform(train[i])
    test[i] = LabelEncoder().fit_transform(test[i])


#결측치 탐색
# print(train.isna().sum())
#결측치 없음.

#이상값 탐색

#생략

#불필요한 컬럼 제거
train = train.drop('ID', axis=1)
test = test.drop('ID', axis=1)

#train_test 분리
from sklearn.model_selection import *
x = train.drop('Reached.on.Time_Y.N', axis=1)
y = train['Reached.on.Time_Y.N']

trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2)

#모델
from sklearn.ensemble import *

model = RandomForestClassifier(max_depth = 5, n_estimators = 600)

param = {"n_estimators" : [100,200,300,400,500], "max_depth" : [3,4,5,6]}

model.fit(trainX,trainY)

pred = model.predict(testX)

#평가지표
from sklearn.metrics import *

print(pred)
print("f1_score",f1_score(testY,pred))
print("Roc_score",roc_auc_score(testY,pred))
print("Auccracy",accuracy_score(testY,pred))
#실제 모델 적용
pred = model.predict_proba(test)
print(pred)

result = pd.DataFrame({"ID":tmp['ID'], "pred":pred[:,1]})
print(result)

result.to_csv("Problem2-1.csv",index=False)