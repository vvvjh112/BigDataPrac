#1-1
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/M2-1.csv')
#
# data = data.head(int(len(data)*0.7))
#
# median1 = round(data['Ozone'].median(),1)
#
# data['Ozone'] = data['Ozone'].fillna(data['Ozone'].mean())
#
# median2 = round(data['Ozone'].median(),1)
#
# answer = round(abs(median1-median2),1)
#
# print(answer)

#7.7


#1-2
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/M2-2.csv')
#
# print(data.isna().sum())
#
# data1 = data[data['HAIR']=="White Hair"]
# data1 = data1[data1['EYE']=="Blue Eyes"]
#
# mini = data1['APPEARANCES'].mean()-(round(data1['APPEARANCES'].std(),2)*1.5)
# maxi = data1['APPEARANCES'].mean()+(round(data1['APPEARANCES'].std(),2)*1.5)
#
# data1 = data1[data1['APPEARANCES']>mini]
# data1 = data1[data1['APPEARANCES']<maxi]
#
# answer = round(data1['APPEARANCES'].mean(),2)
#
# print(answer)
#
# #30.15


#1-3
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/M2-3.csv')
#
# mini = data['Sales'].mean()-(data['Sales'].std()*1.5)
# maxi = data['Sales'].mean()+(data['Sales'].std()*1.5)
#
# train = data[(data['Sales'] > mini) & (data['Sales'] < maxi)]
#
# answer = round(train['Age'].std(),2)
#
# print(answer)
#
# #16.05


#작업형 2유형
# import pandas as pd
# import numpy as np
#
# from sklearn.model_selection import *
# from sklearn.preprocessing import *
# from sklearn.linear_model import *
# from sklearn.metrics import *
#
# train = pd.read_csv('dataset/M2-4-1.csv')
# test = pd.read_csv('dataset/M2-4-2.csv')
#
# print(train.info())
#
# cate_lst = ['vs','am']
#
# def min_max(x):
#     return (x-min(x))/(max(x)-min(x))
#
# for i in cate_lst:
#     train[i] = LabelEncoder().fit_transform(train[i])
#     test[i] = LabelEncoder().fit_transform(test[i])
#
# print(train.isna().sum())
#
# x = train.drop('mpg',axis = 1)
# y = train['mpg']
#
# trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.25)
#
# from sklearn.ensemble import *
# model = RandomForestRegressor(n_estimators = 500, max_depth = 4)
# model.fit(trainX,trainY)
# pred = model.predict(testX)
#
# print("RMSE",mean_squared_error(testY,pred,squared=False))
#
#
#
# pred_1 = model.predict(test)
#
# print(pred_1)
#
# result = pd.DataFrame({"pred":pred_1})
# result.to_csv("exam_2.csv",index = False)


#3-1
# import pandas as pd
# import numpy as np
# import scipy
#
# data = pd.DataFrame({'a':[1,2,3,4,6],'b':[4,5,6,7,8]})
# print(dir(scipy.stats))
# # f_test = f(data['a'],data['b'])
# # print(f_test)

#3-2
# import pandas as pd
# import numpy as np
# from scipy.stats import *
# import scipy.stats
# data = np.array([340,540])
# data1 = np.array([880*0.35,880*0.65])
# # print(dir(scipy.stats))
# # print(help(chisquare))
# result = chisquare(data,data1)
# answer_1 = round(result.statistic,5)
# answer_2 = round(result.pvalue, 5)
# if result.pvalue < 0.05:
#     answer_3 = "기각"
# else:
#     answer_3 = "채택"
#
# print(answer_1)
# print(answer_2)
# print(answer_3)


#1-1
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/M2-1.csv')
# train = data.head(int(len(data)*0.7))
# median1 = round(train['Ozone'].median(),1)
# train.loc[:,'Ozone'] = train['Ozone'].fillna(train['Ozone'].mean())
# median2 = round(train['Ozone'].median(),1)
# answer = round(abs(median1-median2),1)
# print(answer)

#7.7

#1-2
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/M2-2.csv')
#
# data = data[(data['HAIR'] == "White Hair") & (data['EYE'] == "Blue Eyes")]
#
# maxi = round(data['APPEARANCES'].mean(),2)+(round(data['APPEARANCES'].std(),2)*1.5)
# mini = round(data['APPEARANCES'].mean(),2)-(round(data['APPEARANCES'].std(),2)*1.5)
# data = data[(data['APPEARANCES']>mini)&(data['APPEARANCES']<maxi)]
# answer = round(data['APPEARANCES'].mean(),2)
# print(answer)
#30.15


#1-3
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/M2-3.csv')
#
# maxi = data['Sales'].mean()+(data['Sales'].std()*1.5)
# mini = data['Sales'].mean()-(data['Sales'].std()*1.5)
#
# train = data[(data['Sales']<maxi)&(data['Sales']>mini)]
# answer = round(train['Age'].std(),2)
# print(answer)
#16.05

#2
# import pandas as pd
# import numpy as np
#
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import f1_score,accuracy_score,roc_auc_score,mean_squared_error
# from sklearn.ensemble import RandomForestRegressor
#
# train = pd.read_csv('dataset/M2-4-1.csv')
# test = pd.read_csv('dataset/M2-4-2.csv')
#
# print(train.columns)
# print(train.info())
# cate_lst = ['cyl','vs','am']
#
# for i in cate_lst:
#     train[i] = LabelEncoder().fit_transform(train[i])
#     test[i] = LabelEncoder().fit_transform(test[i])
#     train[i] = train[i].astype('category')
#     test[i] = test[i].astype('category')
#
# print(train.head())
#
# x = train.drop('mpg',axis = 1)
# y = train['mpg']
#
# trainX, testX, trainY, testY = train_test_split(x,y,test_size = 0.2)
#
# model = RandomForestRegressor()
#
# model.fit(trainX,trainY)
#
# pred = model.predict(testX)
#
# print("RMSE",mean_squared_error(testY, pred,squared=False))
#
# result = model.predict(test)
#
# answer = pd.DataFrame({'pred':result})
#
# answer.to_csv('Result/모의고사2-1.csv',index=False)

#3-2

# data = pd.DataFrame({'sex':[340,540]})
#
# from scipy.stats import *
#
# exp = [(340+540)*0.35,(340+540)*0.65]
#
# result = chisquare(data['sex'],exp)
#
# answer1 = round(result.statistic,5)
# answer2 = round(result.pvalue,5)
#
# if result.pvalue<0.05:
#     answer3 = "기각"
# else:
#     answer3 = "채택"
#
# print(answer1)
# print(answer2)
# print(answer3)
