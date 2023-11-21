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
import pandas as pd
import numpy as np

data = pd.read_csv('dataset/P220403.csv')

print(data['date_added'].head())

data['date_added'] = pd.to_datetime(data['date_added'],format = '%B %d, %Y')

print(data['date_added'].head())

data = data[(data['date_added'].dt.year == 2018)&(data['date_added'].dt.month == 1)]

data = data[data['country']=='United Kingdom']

answer=(len(data))

print(answer)