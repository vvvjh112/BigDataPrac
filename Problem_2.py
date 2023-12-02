# 2 - 1
# import pandas as pd
# import numpy as np
#
# train = pd.read_csv('dataset/P210204-01.csv')
# test = pd.read_csv('dataset/P210204-02.csv')
# tmp = pd.read_csv('dataset/P210204-02.csv')
#
# #정시 도착여부 예측한 확률
#
# print(train.info())
# print(train.columns)
# #['ID', 'Warehouse_block', 'Mode_of_Shipment', 'Customer_care_calls',
# # 'Customer_rating', 'Cost_of_the_Product', 'Prior_purchases', 'Product_importance',
# # 'Gender', 'Discount_offered', 'Weight_in_gms', 'Reached.on.Time_Y.N']
#
# #라벨인코딩
# from sklearn.preprocessing import *
#
# encoding_ary = ['Warehouse_block','Mode_of_Shipment', 'Product_importance','Gender']
#
# for i in encoding_ary:
#     train[i] = LabelEncoder().fit_transform(train[i])
#     test[i] = LabelEncoder().fit_transform(test[i])
#
#
# #결측치 탐색
# # print(train.isna().sum())
# #결측치 없음.
#
# #이상값 탐색
#
# #생략
#
# #불필요한 컬럼 제거
# train = train.drop('ID', axis=1)
# test = test.drop('ID', axis=1)
#
# #train_test 분리
# from sklearn.model_selection import *
# x = train.drop('Reached.on.Time_Y.N', axis=1)
# y = train['Reached.on.Time_Y.N']
#
# trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2)
#
# #모델
# from sklearn.ensemble import *
#
# model = RandomForestClassifier(max_depth = 5, n_estimators = 600)
#
# param = {"n_estimators" : [100,200,300,400,500], "max_depth" : [3,4,5,6]}
#
# model.fit(trainX,trainY)
#
# pred = model.predict(testX)
#
# #평가지표
# from sklearn.metrics import *
#
# print(pred)
# print("f1_score",f1_score(testY,pred))
# print("Roc_score",roc_auc_score(testY,pred))
# print("Auccracy",accuracy_score(testY,pred))
# #실제 모델 적용
# pred = model.predict_proba(test)
# print(pred)
#
# result = pd.DataFrame({"ID":tmp['ID'], "pred":pred[:,1]})
# print(result)
#
# result.to_csv("Result/Problem2-1.csv",index=False)

#2 - 2
# import pandas as pd
# import numpy as np
# #여행 보험 패키지 가입여부 예측
# train = pd.read_csv('dataset/P210304-01.csv')
# test = pd.read_csv('dataset/P210304-02.csv')
# test_id = test['X']
# # Index(['X', 'Age', 'Employment Type', 'GraduateOrNot', 'AnnualIncome',
# #        'FamilyMembers', 'ChronicDiseases', 'FrequentFlyer',
# #        'EverTravelledAbroad', 'TravelInsurance'],
# #       dtype='object')
#
#
# #전처리
# from sklearn.preprocessing import *
# encoding_ary = ['GraduateOrNot', 'Employment Type', 'FrequentFlyer', 'EverTravelledAbroad','TravelInsurance']
#
# train['TravelInsurance'] = train['TravelInsurance'].astype('category')
# # test['TravelInsurance'] = test['TravelInsurance'].astype('category')
# for i in encoding_ary:
#     train[i] = LabelEncoder().fit_transform(train[i])
#     if (i == 'TravelInsurance'):
#         train[i] = LabelEncoder().fit_transform(train[i])
#     else:
#         test[i] = LabelEncoder().fit_transform(test[i])
#
#
#
# train = train.drop(['X','GraduateOrNot'],axis=1)
# test = test.drop(['X','GraduateOrNot'],axis=1)
#
# print(train.info())
# #결측값 확인
# # print(train.isna().sum())
#
# #train_test_split
# from sklearn.model_selection import *
# x = train.drop('TravelInsurance',axis = 1)
# y = train['TravelInsurance']
#
# trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2)
#
# #모델링 및 평가
# from sklearn.metrics import *
# from sklearn.ensemble import *
#
# model = RandomForestClassifier(n_estimators = 700, max_depth = 5)
#
# model.fit(trainX,trainY)
#
# pred = model.predict(testX)
#
# print("Accuracy",accuracy_score(testY,pred))
# print("roc_auc",roc_auc_score(testY,pred))
# print("f1score",f1_score(testY,pred))
#
#
# print(test.info())
# #결과도출
# pred_proba = model.predict_proba(test)
# tmp = []
# for i in range(0,len(pred_proba)):
#     tmp.append(i)
# result = pd.DataFrame({'index': tmp,'y_pred' : pred_proba[:,1]})
# result.to_csv('Result/Problem2-2.csv',index=False)

#2 - 3
# import pandas as pd
# import numpy as np
#
# train = pd.read_csv('dataset/P220404-01.csv')
# test = pd.read_csv('dataset/P220404-02.csv')
# test_id = test['ID']
#
# # 'ID', 'Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession',
# #        'Work_Experience', 'Spending_Score', 'Family_Size', 'Var_1',
# #        'Segmentation'
#
# ary = ['Gender','Ever_Married','Graduated','Profession','Spending_Score','Var_1']
# from sklearn.preprocessing import *
# train['Segmentation'] = train['Segmentation'].astype('category')
# for i in ary:
#     train[i] = LabelEncoder().fit_transform(train[i])
#     test[i] = LabelEncoder().fit_transform(test[i])
#
# print(train.info())
#
#
# from sklearn.model_selection import *
# from sklearn.ensemble import *
# from sklearn.metrics import *
#
# train = train.drop(['ID'], axis =1 )
# test = test.drop('ID',axis = 1)
#
# x= train.drop('Segmentation',axis =1)
# y = train['Segmentation']
#
# trainX, testX, trainY, testY = train_test_split(x,y,test_size = 0.2)
#
# model = RandomForestClassifier(n_estimators = 600, max_depth = 6)
#
# model.fit(trainX,trainY)
#
# pred = model.predict(testX)
#
# print(pred)
#
#
# print("f1score",f1_score(testY,pred,labels=['A','B','C','D'],average='macro'))
#
# model.fit(x,y)
#
# pred_result = model.predict(test)
#
#
# result = pd.DataFrame({"ID":test_id, "pred" : pred_result})
#
# result.to_csv("Result/Problem_2-3.csv",index=False)

#2 -4
import pandas as pd
import numpy as np

train = pd.read_csv('dataset/P220504-01.csv')
test = pd.read_csv('dataset/P220504-02.csv')
# 'model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']

from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.linear_model import *

# print(train.info())

train['model'] = train['model'].astype('category')
test['model'] = test['model'].astype('category')
Label_lst = ['transmission','fuelType','model']
int_lst = ['mileage','tax']
def min_max(x):
    return (x-min(x))/(max(x)-min(x))

for i in Label_lst:
    train[i] = LabelEncoder().fit_transform(train[i])
    test[i] = LabelEncoder().fit_transform(test[i])

for i in int_lst:
    train[i] = min_max(train[i])
    test[i] = min_max(test[i])

# print(train.info())
#
# print(train.isna().sum())
mm = MinMaxScaler()

x1 = train.drop('price',axis = 1)
y1 = train['price']

train['price'] = mm.fit_transform(train[['price']])
x = train.drop('price',axis = 1)
y = train['price']

trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2)
trainX1, testX1, trainY1, testY1 = train_test_split(x1,y1,test_size=0.2)

model = RandomForestRegressor(n_estimators = 600, max_depth = 4)
model1 = RandomForestRegressor(n_estimators = 600, max_depth = 4)
model.fit(trainX,trainY)
model1.fit(trainX1,trainY1)
pred = model.predict(testX)
pred1 = model1.predict(testX1)


print("Root Mean Squared Error", mean_squared_error(testY,pred,squared = False))
print("Root Mean Squared Error1", mean_squared_error(testY1,pred1,squared = False))


pred = model.predict(test)
pred1 = model1.predict(test)

result1 = pd.DataFrame({"pred":pred1})



result = pd.DataFrame({"pred":pred})
result['pred'] = mm.inverse_transform(result[['pred']])
print(result)
#
result1.to_csv("Result/Problem2-41.csv",index = False)
result.to_csv("Result/Problem2-4.csv",index = False)


#2 - 5
# import pandas as pd
#
# train = pd.read_csv('dataset/P230604-01.csv')
# test = pd.read_csv('dataset/P230604-02.csv')
#
# # 'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
# #        'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
# #        'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
# #        'touch_screen', 'wifi', 'price_range'
# print(train.columns)
# print(train.info())
# # print(train.head())
#
# print(train.describe())
#
# category_lst = ['blue','touch_screen','wifi','dual_sim','four_g','three_g','n_cores']
# min_max_lst = ['battery_power','clock_speed','fc','int_memory','px_height','px_width']
#
# from sklearn.metrics import *
# from sklearn.model_selection import *
# from sklearn.preprocessing import *
# from sklearn.ensemble import *
# # print(train[['talk_time','three_g']].head())
# def min_max(x):
#     return (x-min(x))/(max(x)-min(x))
# train['price_range'] = train['price_range'].astype('category')
# for i in category_lst:
#     train[i] = train[i].astype('category')
#     test[i] = test[i].astype('category')
# x = train.drop('price_range',axis =1)
# y = train['price_range']
#
# # x = pd.get_dummies(x)
#
# trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2, random_state = 2051)
#
# model = RandomForestClassifier(random_state = 2051)
#
# model.fit(trainX,trainY)
#
# pred = model.predict(testX)
#
# print('Accuracy',accuracy_score(testY,pred))
#
# print('macro_f1',f1_score(testY,pred, labels = [0,1,2,3],average = 'macro'))
#
# pred = model.predict(test.drop('id',axis=1))
#
# result = pd.DataFrame({"pred":pred})
#
# result.to_csv("Result/Problem2-5.csv",index = False)
