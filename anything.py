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



#3-1

# import pandas as pd
# import numpy as np
# from scipy.stats import *
#
# data = pd.read_csv('dataset/P230605.csv',encoding = 'euc-kr')
#
# answer1 = round(len(data[data['코드']==4])/len(data),3)
#
# print(answer1)
# tb = data.groupby([data['코드']]).size()
# tb = tb.reset_index(name = '개수')
# leng = len(data)
# dt = pd.DataFrame({'기대값':[0.05*leng,0.1*leng,0.05*leng,0.8*leng]})
#
# print(tb['개수'],dt['기대값'])
# result = chisquare(tb['개수'],dt['기대값'])
# print(result)

#3-2
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from scipy.stats import *
#
# data = pd.read_csv('dataset/P230606.csv')
#
# trainX = data[['O3','Solar','Wind']]
# trainY = data['Temperature']
#
# model = LinearRegression()
# model.fit(trainX, trainY)
#
# coef = model.coef_[0]
# answer1 = round(coef,3)
# print(answer1)
# import scipy.stats
# # print(dir(scipy.stats))
# # print(len(data))
# print(levene(data['Wind'],data['Temperature']))
# result = ttest_ind(data['Wind'],data['Temperature'])
# print(round(result.pvalue,3))


#1-3-1
# import pandas as pd
# import numpy as np
# from scipy.stats import *
#
# data = pd.DataFrame({'before':[200,210,190,180,175],'after':[180,175,160,150,160]})
#
# result = ttest_rel(data['after'],data['before'],alternative = 'less')
#
# print(result)


# �
# ˉ
# X
# ˉ
#  는 표본 평균 (51.5),
# �
# μ는 모평균 (50),
# �
# σ는 모표준편차 (알려져 있지 않으므로 표본 표준편차인 2.5를 사용),
# �
# n은 표본 크기 (25)입니다.

# import numpy as np
# from scipy import stats
#
# # 데이터
# data = np.array([52, 50, 49, 53, 51, 52, 50, 52, 51, 53, 51, 50, 51, 50, 49, 53, 52, 51, 50, 52, 49, 51, 53, 50, 52])
#
# # 표본 평균, 표본 크기, 표본 표준편차 계산
# sample_mean = np.mean(data)
# sample_size = len(data)
# sample_std = np.std(data, ddof=1)  # ddof=1은 비편향 표본 표준편차를 의미합니다.
#
# # 모평균 가정
# population_mean = 50
#
# # Z-검정 통계량 계산
# z_stat = (sample_mean - population_mean) / (sample_std / np.sqrt(sample_size))
#
# # 양측 검정이므로, Z-검정 임계값은 약 ±1.96
# critical_value = 1.96
#
# # P-value 계산
# p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
#
# # 결과 출력
# print(f"Z-검정 통계량: {z_stat}")
# print(f"P-value: {p_value}")
#
# # 유의수준 0.05에서의 결정
# if abs(z_stat) > critical_value:
#     print("귀무가설을 기각합니다.")
# else:
#     print("귀무가설을 기각하지 않습니다.")



# import pandas as pd
# import numpy as np
#
# from scipy.stats import *
#
# a = np.array([1,2,3,4,6])
# b = np.array([4,5,6,7,8])
#
# def f_test(x,y):
#     if np.var(x,ddof = 1) < np.var(x,ddof = 1):
#         x,y = y,x
#
#     f_value = np.var(x,ddof = 1)/np.var(y,ddof = 1)
#     p_value = (1-f.cdf(f_value,x.size-1,y.size-1))*2
#     return f_value,p_value
#
# stat, p = f_test(a,b)
#
# print(round(stat,2))
# print(round(p,4))
# if(p<0.05):
#     print("대립가설채택")
# else:
#     print("귀무가설채택")



#
# import pandas as pd
# import numpy as np
#
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import *
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
#
# train= pd.read_csv('dataset/P220404-01.csv')
# test = pd.read_csv('dataset/P220404-02.csv')
#
# print (train.columns)
#
# train = train.drop('ID',axis = 1)
# test = test.drop('ID',axis =1)
#
# print(train.info())
#
# cate_lst=['Gender','Ever_Married','Graduated', 'Profession','Spending_Score','Spending_Score','Var_1']
# train['Segmentation'] = train['Segmentation'].astype('category')
#
# for i in cate_lst:
#     train[i] = LabelEncoder().fit_transform(train[i])
#     test[i] = LabelEncoder().fit_transform(test[i])
#     train[i] = train[i].astype('category')
#     test[i] = test[i].astype('category')
#
# print(train.isna().sum())
# print(train.info())
#
# x = train.drop('Segmentation',axis = 1)
# y = train['Segmentation']
#
# trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2,random_state = 2000)
#
# model = RandomForestClassifier(n_estimators = 600 ,max_depth = 8,random_state=2000)
#
# model.fit(trainX,trainY)
#
# pred = model.predict(testX)
#
# print("F1_score",f1_score(testY,pred, labels = ['A','B','C','D'], average='macro'))
# print("accuracy",accuracy_score(testY,pred))
# # print("rcoauc",roc_auc_score(testY,pred))


#mean(x)-표본평균 / (표준편차/(np.sqrt(표본개수))
#pvalue = (1-norm.cdf(위에거))*2


# #
# import pandas as pd
# import numpy as np
# data = pd.read_csv('dataset/P210201.csv')
#
# top10 = data.sort_values('crim',ascending = False).head(10)
# print(top10.head(10))
# top10 = top10['crim'].iloc[9]
# print(top10)
#
# def change(x):
#     if x>top10:
#         return top10
#     else:
#         return x
#
# data['crim'] = data['crim'].apply(change)
# data = data[data['age']>80]
# print(round(data['crim'].mean(),2))


# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210302.csv')
#
# tmp = data.isna().sum()/len(data)
# print(tmp)
# tmp = tmp.sort_values(ascending = False)
# result = tmp.head(1)
# answer = result.index[0]
# print(answer)

# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210303.csv')
#
# data = data.dropna(subset=['country','year','new_sp'])
# data = data[data['year']==2000]
# mean1 = data['new_sp'].mean()
#
# print(mean1)
# answer = len(data[data['new_sp']>mean1])
# print(int(answer))


# import pandas as aspd
# import numpy as np
# data = pd.read_csv('dataset/P220401.csv')
#
# percen3 = np.percentile(data['y'],75)
# percen1 = np.percentile(data['y'],25)
#
# answer = int(abs(percen3-percen1))
# print(answer)

# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P220403.csv')
# data['date_added'] = pd.to_datetime(data['date_added'])
# print(data['date_added'])
# answer = len(data[(data['date_added'].dt.year == 2018) & (data['date_added'].dt.month == 1) & (data['country'] == "United Kingdom")])
# print(answer)

# import pandas as pd
# data = pd.read_csv('dataset/P230601.csv')
#
# data['신고일시'] = pd.to_datetime(data['신고일시'])
# data['출동일시'] = pd.to_datetime(data['출동일시'])
# data['소요시간'] = (data['출동일시'] - data['신고일시']).dt.total_seconds()
#
# print(data.head())
#
# gp = data.groupby([data['출동소방서'],data['신고일시'].dt.year, data['신고일시'].dt.month]).mean('소요시간')
#
# print(gp.head(99))
#
# gp = gp.sort_values('소요시간',ascending = False).head(1)
#
# result = gp['소요시간'].iloc[0]/60
#
# answer = round(result,0)
# print(answer)

# import pandas as pd
# data = pd.read_csv('dataset/P230603.csv')
# print(data)
#
# data['sum'] = data['강력범']+data['절도범']+data['폭력범']   +data['지능범']  +data['풍속범']+data ['기타형사범']
# data['년월'] = pd.to_datetime(data['년월'])

# gp = data.groupby([data['년월'].dt.year]).mean('sum').sort_values('sum',ascending = False).head(1)
# print(gp['sum'].iloc[0])


# import numpy as np
# from scipy.stats import *
#
# x = np.array([25,27,31,23,24,30,26])
#
# # z = (np.mean(x)-모평균) / 표준편차 * np.sqrt(np.s\ize(x))
#
# p = (1-norm.cdf(z))*2
#
# print(z,p)


import numpy as np
from scipy.stats import f
df1 = np.array([1,2,3,4,6])
df2 = np.array([4,5,6,7,8])
def f_test(x,y):
    if np.var(x,ddof =1) < np.var(y,ddof =1):
        x,y = y, x
    f1 = np.var(x,ddof = 1) / np.var(y,ddof = 1)
    p = (1 - f.cdf(f1,np.size(x)-1, np.size(y)-1))*2
    return f1,p

print(f_test(df1,df2))