
#1-1
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/M1-1.csv')
#
# train = data.head(int(len(data)*0.7))
#
# result = train.sort_values('price',ascending = False).head(5)
#
# answer = int(result['depth'].median())
#
# print(answer)
#
# #62

#1-2
# import pandas as pd
# import numpy as np
# data = pd.read_csv('dataset/M1-2.csv')
# print(data.columns)
# # print(data['TotalCharges'])
# # temp = data[data['TotalCharges'] != '']
# data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors='coerce')
# # print(data.head())
# data['TotalCharges']= data['TotalCharges'].astype(float)
# data = data.dropna(subset=['TotalCharges'])
# print(data.isna().sum())
#
#
# cut = (data['TotalCharges'].std())*1.5
# mean = data['TotalCharges'].mean()
#
# maxi = mean+cut
# mini = mean-cut
#
# data = data[data['TotalCharges']<=maxi]
# data = data[data['TotalCharges']>=mini]
#
# answer = int(data['TotalCharges'].mean())
# print(answer)
# #1663


#1-3
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/M1-3.csv')
#
# set1 = data[data['am']==1]
# sort1 = set1.sort_values('hp',ascending = True)
# sort1 = sort1.head(5)
# mean1 = sort1['mpg'].mean()
#
# set2 = data[data['am']==0]
# sort2 = set2.sort_values('hp',ascending = True)
# sort2 = sort2.head(5)
# mean2 = sort2['mpg'].mean()
#
# answer=round(abs(mean1-mean2),1)
# print(answer)

#작업형2
# import pandas as pd
# import numpy as np
# train = pd.read_csv('dataset/M1-4-1.csv')
# test = pd.read_csv('dataset/M1-4-2.csv')
# # print(test.head())
#
# # print(train.columns)
# # print(train.info())
# cate_lst = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
# del_lst = ['PaymentMethod', 'InternetService','PaperlessBilling','tenure','gender','StreamingMovies']
# scaling_lst = ['MonthlyCharges','TotalCharges']
#
# from sklearn.preprocessing import *
# for i in cate_lst:
#     # if i != 'Churn':
#     test[i] = LabelEncoder().fit_transform(test[i])
#     train[i] = LabelEncoder().fit_transform(train[i])
#     train[i] = train[i].astype('category')
#     test[i] = test[i].astype('category')
# print(test.isna().sum())
# for i in del_lst:
#     train=train.drop(i,axis=1)
#     test=test.drop(i,axis=1)
#
# def min_max(x):
#     return (x-min(x))/(max(x)-min(x))
#
# for i in scaling_lst:
#     train[i] = min_max(train[i])
#     test[i] = min_max(test[i])
#
# # print(train.isna().sum())
# #
# # print(train.info())
# # print(train.head())
#
# from sklearn.model_selection import *
# from sklearn.ensemble import *
# from sklearn.linear_model import *
# from sklearn.metrics import *
#
# model = RandomForestClassifier(n_estimators = 500, max_depth=6)
#
# x = train.drop(['Churn'],axis=1)
# y = train['Churn']
# test = test.drop('Churn',axis=1 )
# trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.20)
#
# model.fit(trainX,trainY)
#
# tmp_pred = model.predict(testX)
# # print(tmp_pred)
# print("Accuracy",accuracy_score(testY,tmp_pred))
# print("f1",f1_score(testY,tmp_pred))
# print("Roc_auc",roc_auc_score(testY,tmp_pred))
# # print(test.isna().sum())
#
# def yes_or_no(x):
#     if x:
#         return 'No'
#     else:
#         return 'Yes'
#
# pred = model.predict(test)
#
# result = pd.DataFrame({"pred":pred})
# result['pred'] = result['pred'].apply(yes_or_no)
#
# # print(result)
#
# result.to_csv("Result/exam2-1.csv",index=False)

#3-1
# import pandas as pd
# import numpy as np
#
# data = pd.DataFrame({'before':[200,210,190,180,175],'after':[180,175,160,150,160]})
#
# from scipy.stats import *
# result = ttest_rel(data['after'],data['before'],alternative = 'less')
#
# print(result)

#3-2
# import pandas as pd
# import numpy as np
# from scipy.stats import *
#
# data = pd.read_csv('dataset/M1-6.csv')
#
# a = shapiro(data['Temp'])
# print(a)
# # print(help(ttest_1samp))
# # result = ttest_1samp(data['Temp'],popmean=75)
#
# result = wilcoxon(data['Temp'])
# print(result)



#1-1
import pandas as pd
import numpy as np

data = pd.read_csv('dataset/M1-1.csv')

train = data.head(int(len(data)*0.7))

result = train.sort_values('price',ascending = False).head(5)

answer = int(result['depth'].median())

print(answer)

#61

#1-2
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/M1-2.csv')
#
# data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors = "coerce")
# data = data.dropna(subset=['TotalCharges'])
#
# print(data.isna().sum())
#
# maxi = data['TotalCharges'].mean()+(data['TotalCharges'].std()*1.5)
# mini = data['TotalCharges'].mean()-(data['TotalCharges'].std()*1.5)
#
# result = data[(data['TotalCharges']>mini)&(data['TotalCharges']<maxi)]
#
# answer = int(result['TotalCharges'].mean())
#
# print(answer)

#1663

#1-3
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/M1-3.csv')
#
# set1 = data[data['am']==1].sort_values('hp',ascending = True).head(5)
# set2 = data[data['am']==0].sort_values('hp',ascending = True).head(5)
#
# result1 = set1['mpg'].mean()
# result2 = set2['mpg'].mean()
#
# answer = round(abs(result1-result2),1)
#
# print(answer)
#8.4


#2




#3-1
# import pandas as pd
# import numpy as np
#
# from scipy.stats import *
#
# data = pd.DataFrame({'before':[200,210,190,180,175],'after':[180,175,160,150,160]})
#
# result = ttest_rel(data['after'],data['before'],alternative = 'less')
#
# answer1 = round(result.statistic,5)
# answer2 = round(result.pvalue,5)
# if result.pvalue<0.05:
#     answer3 = "기각"
# else:
#     answer3= "채택"
# print(answer1)
# print(answer2)
# print(answer3)
# -7.0763
# 0.00105
# 기각


#3-2
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/M1-6.csv')
# from scipy.stats import *
# import scipy.stats
# # print(dir(scipy.stats))
# test = shapiro(data['Temp'])
# print(test)
# result = wilcoxon(data['Temp'])
# answer1 = round(result.statistic,3)
# answer2 = round(result.pvalue,3)
#
# if result.pvalue<0.05:
#     answer3 = "기각"
# else:
#     answer3= "채택"
#
# print(answer1)
# print(answer2)
# print(answer3)

# 3.767
# 0.0
# 기각