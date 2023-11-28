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
import pandas as pd
import numpy as np

data = pd.read_csv('dataset/P230601.csv')



data['신고일시'] = pd.to_datetime(data['신고일시'])
data['출동일시'] = pd.to_datetime(data['출동일시'])

data['소요시간'] = data['출동일시']-data['신고일시']
data['소요시간'] = data['소요시간'].dt.total_seconds()

# print(data.head())

group = data.groupby([data['출동소방서'],data['신고일시'].dt.year, data['신고일시'].dt.month]).mean('소요시간')

# print(group.head())

result = group.sort_values('소요시간',ascending = False).head(1)
answer = result['소요시간'].iloc[0]/60
print(int(round(answer,0)))