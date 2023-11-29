#1-1
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210201.csv')
#
# t10 = data.sort_values('crim',ascending = False).head(10)
#
# t10 = t10['crim'].iloc[9]
#
# def change(x):
#     if (x>t10):
#         return t10
#     else:
#         return x
#
# data['crim'] = data['crim'].apply(change)
#
# result = data[data['age']>80]
# answer = round(result['crim'].mean(),2)
# print(answer)

#1-2
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210202.csv')
#
# train = data.head(int(len(data)*0.8))
#
# result1 = train['total_bedrooms'].std()
#
# train.loc[:,'total_bedrooms'] = train['total_bedrooms'].fillna(train['total_bedrooms'].median())
# result2 = train['total_bedrooms'].std()
#
# answer = round(abs(result1-result2),2)
# print(answer)

#1-3
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210203.csv')
#
# maxi = data['charges'].mean()+(data['charges'].std()*1.5)
# mini = data['charges'].mean()-(data['charges'].std()*1.5)
#
# result = data[(data['charges']>maxi)|(data['charges']<mini)]
# answer = int(result['charges'].sum())
# print(answer)

#1-4
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210301.csv')
# data = data.dropna()
# train = data.head(int(len(data)*0.7))
# answer = int(np.percentile(train['housing_median_age'],25))
# print(answer)


#1-5
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210302.csv')
#
# result = data.isna().sum()/len(data)
# result = result.sort_values(ascending= False).head(1)
# print(result)
# answer = result.index[0]
# print(answer)

#1-6
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210303.csv')
#
# data = data.dropna(subset=['country','year','new_sp'])
#
# # print(data['year'].head())
#
# tmp = data[data['year']==2000]
# group = tmp.groupby(tmp['country'])
# mean = tmp['new_sp'].sum()/len(group)
# print(mean)
#
# result = tmp[tmp['new_sp']>mean]
# answer = round(len(result),2)
# print(answer)


#1-9
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P220403.csv')
# print(data.columns)
# data['date_added'] = pd.to_datetime(data['date_added'])
# print(data['date_added'].head(99))
# result = data[(data['date_added'].dt.year == 2018) & (data['date_added'].dt.month == 1) & (data['country']== "United Kingdom")]
# answer = int(len(result))
# print(answer)


#1-13
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P230601.csv')
#
#
#
# data['신고일시'] = pd.to_datetime(data['신고일시'])
# data['출동일시'] = pd.to_datetime(data['출동일시'])
#
# data['소요시간'] = data['출동일시']-data['신고일시']
# data['소요시간'] = data['소요시간'].dt.total_seconds()
#
# # print(data.head())
#
# group = data.groupby([data['출동소방서'],data['신고일시'].dt.year, data['신고일시'].dt.month]).mean('소요시간')
#
# # print(group.head())
#
# result = group.sort_values('소요시간',ascending = False).head(1)
# answer = result['소요시간'].iloc[0]/60
# print(int(round(answer,0)))

#1-15
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P230603.csv')
# data['sum'] = data['강력범']+data['절도범']+data['폭력범']+data['지능범']+data['풍속범']+data['기타형사범']
# data['년월'] = pd.to_datetime(data['년월'])
# group = data.groupby([data['년월'].dt.year]).mean('sum')
# tmp = group.sort_values('sum',ascending = False).head(1)
# result = tmp['sum'].iloc[0]
# answer = int(result)
# print(answer)


#2-1
# import pandas as pd
# import numpy as np
#
# train = pd.read_csv('dataset/P210204-01.csv')
# test = pd.read_csv('dataset/P210204-02.csv')
# test_id = test['ID']
#
# train = train.drop('ID',axis = 1)
# test = test.drop('ID',axis = 1)
# print(train.columns)
#
# cate_lst = ['Warehouse_block', 'Mode_of_Shipment','Product_importance', 'Gender']
# train['Reached.on.Time_Y.N'] = train['Reached.on.Time_Y.N'].astype('category')
#
# from sklearn.preprocessing import LabelEncoder
#
# for i in cate_lst:
#     train[i] = LabelEncoder().fit_transform(train[i])
#     test[i] = LabelEncoder().fit_transform(test[i])
#     train[i] = train[i].astype('category')
#     test[i] = test[i].astype('category')
#
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import *
# from sklearn.ensemble import RandomForestClassifier
#
# print(train.info())
# x = train.drop('Reached.on.Time_Y.N',axis =1)
# y = train['Reached.on.Time_Y.N']
#
# model = RandomForestClassifier(max_depth = 8,random_state=2000)
#
# trainX, testX, trainY, testY = train_test_split(x,y,test_size = 0.1,random_state = 2000)
#
# model.fit(trainX,trainY)
#
# pred = model.predict(testX)
#
# print("Accuarcy",accuracy_score(testY,pred))
# print("f1_score", f1_score (testY,pred))
# print("Roc_Auc",roc_auc_score(testY,pred))
#
# result = model.predict_proba(test)
# print(result)
# answer = pd.DataFrame({'pred':result[:,1]})
# print(answer.head())

#2-2
# import pandas as pd
# import numpy as np
# from sklearn.metrics import *
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
#
# train = pd.read_csv('dataset/P210304-01.csv')
# test = pd.read_csv('dataset/P210304-02.csv')
# test_X = test['X']
# train = train.drop('X',axis = 1)
# test = test.drop('X',axis = 1)
#
# print(train.columns)
# print(train.head())
# cate_lst = ['Employment Type', 'GraduateOrNot','FrequentFlyer', 'EverTravelledAbroad']
# train['TravelInsurance'] = train['TravelInsurance'].astype('category')
#
# print(train.info())
# lb = LabelEncoder()
# for i in cate_lst :
#     train[i] = lb.fit_transform(train[i])
#     test[i] = lb.fit_transform(test[i])
#     train[i] = train[i].astype('category')
#     test[i] = test[i].astype('category')
#
# print(train.isna().sum())
#
# x = train.drop('TravelInsurance',axis = 1)
# y = train['TravelInsurance']
#
# trainX, testX, trainY, testY = train_test_split(x,y,test_size = 0.2, random_state = 2000)
#
# model = RandomForestClassifier(max_depth = 8,random_state = 2000)
#
# model.fit(trainX, trainY)
#
# pred= model.predict(testX)
#
# print("f1_score",f1_score(testY,pred))
# print("Roc_Auc",roc_auc_score(testY,pred))
# print("Accuracy",accuracy_score(testY,pred))

# result = model.predict(test)
# answer = pd.DataFrame({'pred':result})
# print(answer)