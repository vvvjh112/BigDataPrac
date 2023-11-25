# 1-1
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210201.csv')
#
# data = data.sort_values('crim',ascending = False)
# tmp = data.head(10)
# # print(tmp)
#
# top10 = tmp['crim'].iloc[9]
#
# def cutline(x):
#     if x>top10:
#         return top10
#     else:
#         return x
#
# data['crim'] = data['crim'].apply(cutline)
# print(data)
# result = data[data['age']>=80]
# answer = round(result['crim'].mean(),2)
# print(answer)

#1-2
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210202.csv')
# data = data.head(int(len(data)*0.8))
# result1 = data['total_bedrooms'].std()
# data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].median())
# result2 = data['total_bedrooms'].std()
#
# answer = abs(round(result1-result2,2))
# print(answer)

# 1-3

# import pandas as pd
#
# data = pd.read_csv('dataset/P210203.csv')
#
# maxi = data['charges'].mean()+(data['charges'].std()*1.5)
# mini = data['charges'].mean()-(data['charges'].std()*1.5)
#
# result1 = data[data['charges']<=mini].sum()
# result2 = data[data['charges']>=maxi].sum()
#
# answer = int(result1['charges']+result2['charges'])
# print(answer)


#1-4
# import pandas as pd
#
# data = pd.read_csv('dataset/P210301.csv')
# data = data.dropna()
# data = data.head(int(len(data)*0.7))
# import numpy as np
# answer = int(np.percentile(data['housing_median_age'],25))
# print(answer)

#1-5
# import pandas as pd
# import numpy as np
#
# data =pd.read_csv('dataset/P210302.csv')
#
# tmp = data.isna().sum()/len(data)
# tmp = tmp.sort_values(ascending = False)
# answer = tmp.index[0]
# print(answer)

#1-6
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210303.csv')
#
# data = data.dropna(subset=['country','year','new_sp'])
# data = data[data['year']==2000]
#
# group = data['new_sp'].groupby(data['country']).mean()
#
# cut = round(group.mean(),2)
#
# group = group[group>=cut]
#
# answer = len(group)
#
# print(answer)

#1-7
#
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P220401.csv')
#
# result1 = np.percentile(data['y'],25)
# result2 = np.percentile(data['y'],75)
#
# answer=(int(result2-result1))
#
# print(answer)

#1-8
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P220402.csv')
#
# data['ratio'] = (data['num_loves']+data['num_wows'])/data['num_reactions']
#
# answer = len(data[(data['ratio']>0.4) & (data['ratio']<0.5)])
#
# print(answer)

#1-9
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P220403.csv')
#
# print(data['date_added'].head())
#
# data['date_added'] = pd.to_datetime(data['date_added'],format = '%B %d, %Y')
#
# print(data['date_added'].head())
#
# data = data[(data['date_added'].dt.year == 2018)&(data['date_added'].dt.month == 1)]
#
# data = data[data['country']=='United Kingdom']
#
# answer=(len(data))
#
# print(answer)

# 1-10
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P220501.csv')
#
# print(data.columns)
#
# data = data[(data['종량제봉투종류'] == '규격봉투') & (data['종량제봉투용도']=='음식물쓰레기')&(data['2L가격']>0)]
#
# result = data['2L가격']
#
# answer = int(round(result.mean()))
# print(answer)

#1 -11
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P220502.csv')
#
# def bmi(w,h):
#     return w/((h/100)**2)
#
# data['bmi'] = bmi(data['Weight'],data['Height'])
# #정상 - 위험 절댓값
# normal = len(data[(data['bmi']>=18.5) & (data['bmi']<23)])
# warning = len(data[(data['bmi']>=23) & (data['bmi']<25)])
#
# answer = int(round(abs(normal-warning),0))
# print(answer)

#1-12
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P220503.csv')
#
# print(data.columns)
#
# data['sun'] = data['전입학생수(계)']-data['전출학생수(계)']
#
# result = data.sort_values('sun',ascending = False).head(1)
#
# answer = (result['전체학생수(계)'].iloc[0])
#
# print(answer)

#1-13
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P230601.csv')
#
# data['신고일시'] = pd.to_datetime(data['신고일시'])
# data['출동일시'] = pd.to_datetime(data['출동일시'])
# data['소요시간'] = (data['출동일시']-data['신고일시']).dt.total_seconds()

# group = data.groupby([data['출동소방서'],data['신고일시'].dt.year,data['신고일시'].dt.month]).mean('소요시간')
# group = group.sort_values('소요시간',ascending = False).head(1)
# result = group['소요시간'].iloc[0]/60
# answer = int(round(result,0))
# print(answer)

#1-14
# import pandas as pd
# # import numpy as np
# #
# # data = pd.read_csv('dataset/P230602.csv')
# # data['sum'] = data['student_1']+ data['student_2']+data['student_3']+data['student_4']+data['student_5']+data['student_6']
# # data['ratio'] = data['sum'] / data['teacher']
# # data = data.sort_values('ratio',ascending = False).head(1)
# # answer = data['teacher'].iloc[0]
# # print(answer)


#1-5
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P230603.csv')
# print(data.columns)
# data['년월'] = pd.to_datetime(data['년월'])
# data['합계'] = data['강력범'] + data['절도범'] + data['폭력범'] + data['지능범'] + data['풍속범'] + data['기타형사범']
# group = data.groupby([data['년월'].dt.year]).sum('합계')
# group = group.sort_values('합계',ascending = False).head(1)
# data = data[data['년월'].dt.year == 2013]
# answer = int(round(data['합계'].mean(),0))
# print(answer)




##############################################################

#2-1
# import pandas as pd
# import numpy as np
#
# from sklearn.preprocessing import *
# from sklearn.model_selection import *
# from sklearn.ensemble import *
# from sklearn.metrics import *
#
# train = pd.read_csv('dataset/P210204-01.csv')
# test = pd.read_csv('dataset/P210204-02.csv')
# test_id = test['ID']
# test = test.drop('ID',axis = 1)
# print(train.info())
#
# cate_lst = ['Warehouse_block','Mode_of_Shipment','Product_importance','Gender']
#
# train['Reached.on.Time_Y.N'] = train['Reached.on.Time_Y.N'].astype('category')
# for i in cate_lst :
#     train[i] = LabelEncoder().fit_transform(train[i])
#     test[i] = LabelEncoder().fit_transform(test[i])
#     train[i] = train[i].astype('category')
#     test[i] = test[i].astype('category')
#
# print(train.isna().sum())
# print(test.isna().sum())
#
# x = train.drop(['ID','Reached.on.Time_Y.N'],axis = 1)
# y = train['Reached.on.Time_Y.N']
# trainX, testX, trainY, testY = train_test_split(x,y,test_size = 0.2)
#
# model = RandomForestClassifier(n_estimators = 500, max_depth = 6)
#
# model.fit(trainX,trainY)
#
# pred = model.predict(testX)
#
# print("Accuracy",accuracy_score(testY,pred))
# print("f1score",f1_score(testY,pred))
# print("Roc_Auc",roc_auc_score(testY,pred))
#
# result = model.predict_proba(test)
#
# tocsv = pd.DataFrame({'ID' : test_id ,'pred':result[:,1]})
#
# tocsv.to_csv("Result/수험번호2-1.csv",index = False)


# 2 -2
# import pandas as pd
# import numpy as np
#
# from sklearn.metrics import *
# from sklearn.preprocessing import *
# from sklearn.model_selection import *
# from sklearn.ensemble import *
#
# train = pd.read_csv('dataset/P210304-01.csv')
# test = pd.read_csv('dataset/P210304-02.csv')
# test_X = test['X']
# test = test.drop('X',axis = 1)
# print(train.info())
#
# cate_lst = ['Employment Type','GraduateOrNot','FrequentFlyer','EverTravelledAbroad']
# train['TravelInsurance'] = train['TravelInsurance'].astype('category')
# for i in cate_lst:
#     train[i] = LabelEncoder().fit_transform(train[i])
#     test[i] = LabelEncoder().fit_transform(test[i])
#     train[i] = train[i].astype('category')
#     test[i] = test[i].astype('category')
#
# print(train.isna().sum())
# print(test.isna().sum())
#
# x = train.drop(['X','TravelInsurance'], axis = 1)
# y = train['TravelInsurance']
#
# trainX, testX, trainY, testY = train_test_split(x,y,test_size = 0.2)
#
# model = RandomForestClassifier(n_estimators = 400, max_depth = 5)
#
# model.fit(trainX,trainY)
#
# pred = model.predict(testX)
#
# print("Accuracy",accuracy_score(testY,pred))
# print("f1_score",f1_score(testY,pred))
# print("roc_auc",roc_auc_score(testY,pred))
#
# result = model.predict_proba(test)
#
# csv = pd.DataFrame({'index':test_X,'y_pred':result[:,1]})
# csv.to_csv("Result/수험번호2-2.csv",index = False)

#2-3
# import pandas as pd
# import numpy as np
# from sklearn.metrics import *
# from sklearn.preprocessing import *
# from sklearn.model_selection import *
# from sklearn.ensemble import *
#
# train = pd.read_csv('dataset/P220404-01.csv')
# test = pd.read_csv('dataset/P220404-02.csv')
# test_id = test['ID']
# test= test.drop('ID',axis = 1)
#
# print(train.info())
#
# cate_lst= ['Gender','Ever_Married','Graduated','Profession','Spending_Score','Var_1','Segmentation']
#
# for i in cate_lst:
#     if i == 'Segmentation':
#         # train[i] = LabelEncoder().fit_transform(train[i])
#         train[i] = train[i].astype('category')
#     else:
#         train[i] = LabelEncoder().fit_transform(train[i])
#         train[i] = train[i].astype('category')
#         test[i] = LabelEncoder().fit_transform(test[i])
#         test[i] = test[i].astype('category')
# train['Spending_Score'] = minmax_scale(train['Spending_Score'])
# test['Spending_Score'] = minmax_scale(test['Spending_Score'])
# print(train.isna().sum())
# print(test.isna().sum())
#
# x = train.drop(['ID','Segmentation'],axis = 1)
# y = train['Segmentation']
#
# trainX, testX, trainY, testY = train_test_split(x,y,test_size= 0.2)
#
# model = RandomForestClassifier(n_estimators = 600, max_depth = 6)
#
# model.fit(trainX,trainY)
#
# pred = model.predict(testX)
#
# print("Accuracy",accuracy_score(testY,pred))
# # print("roc_auc",roc_auc_score(testY,pred))
# print("f1score",f1_score(testY,pred,labels=['A','B','C','D'],average = 'macro'))

#2-4
# import pandas as pd
# import numpy as np
#
# train = pd.read_csv('dataset/P220504-01.csv')
# test = pd.read_csv('dataset/P220504-02.csv')
#
# from sklearn.preprocessing import *
# from sklearn.metrics import *
# from sklearn.model_selection import *
# from sklearn.ensemble import *
#
# def minmax(x):
#     return (x-min(x))/(max(x)-min(x))
# train['mileage'] = minmax(train['mileage'])
# test['mileage'] = minmax(test['mileage'])
#
# cate_lst = ['engineSize','model','transmission','fuelType']
# for i in cate_lst:
#     train[i] = LabelEncoder().fit_transform(train[i])
#     test[i] = LabelEncoder().fit_transform(test[i])
#     train[i] = train[i].astype('category')
#     test[i] = test[i].astype('category')
#
# train['price'] = train['price'].astype(int)
#
# print(train.info())
#
# print(train.head())
#
# print(train.isna().sum())
# print(test.isna().sum())
#
# x = train.drop('price',axis = 1)
# y = train['price']
#
# trainX, testX, trainY, testY = train_test_split(x,y,test_size = 0.2)
#
# from sklearn.linear_model import *
# model1 = RandomForestRegressor()
# model2 = LinearRegression()
#
# model1.fit(trainX,trainY)
# model2.fit(trainX,trainY)
#
# pred1 = model1.predict(testX)
# pred2 = model2.predict(testX)
#
# print("RMSE",mean_squared_error(testY,pred1,squared = False)," ",mean_squared_error(testY,pred2,squared=False))
#
# result = model1.predict(test)
#
# answer = pd.DataFrame({'pred':result})
#
# answer.to_csv("Result/수험번호2-4.csv",index= False)

#2-5
# import pandas as pd
# import numpy as np
#
# train = pd.read_csv('dataset/P230604-01.csv')
# test = pd.read_csv('dataset/P230604-02.csv')
#
# print(train.columns)
#
# from sklearn.preprocessing import *
# from sklearn.model_selection import *
# from sklearn.metrics import *
# from sklearn.ensemble import *
#
# cate_lst = ['blue','dual_sim','four_g','int_memory','three_g','touch_screen','wifi']
# train['price_range'] = train['price_range'].astype('category')
# for i in cate_lst:
#     train[i] = train[i].astype('category')
#     test[i] = test[i].astype('category')
#     train[i] = LabelEncoder().fit_transform(train[i])
#     test[i] = LabelEncoder().fit_transform(test[i])
#
# print(train.isna().sum())
#
# x = train.drop('price_range',axis = 1)
# y = train['price_range']
#
# trainX, testX, trainY, testY = train_test_split(x,y,test_size = 0.3)
#
# model = RandomForestClassifier()
#
# model.fit(trainX,trainY)
#
# pred = model.predict(testX)
#
#
# print("f1_score",f1_score(testY,pred,labels = [0,1,2,3], average = 'macro'))
# print("Accuracy",accuracy_score(testY,pred))
#
# test_id = test['id']
# test = test.drop('id',axis = 1)
#
# result = model.predict(test)
#
# answer = pd.DataFrame({ 'id' : test_id,'pred' : result})
#
# answer.to_csv("Result/수험번호2-5.csv",index = False)


#3-1
import pandas as pd
import numpy as np
from scipy.stats import *

data = pd.read_csv('dataset/P230605.csv',encoding = "euc_kr")

answer_1 = round((len(data[data['코드']==4])/len(data)),3)
print(answer_1)

print(data)