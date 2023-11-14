

#작업형 1 - 1
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210201.csv')
#
# top10 = data['crim'].sort_values(ascending=False).head(10)
#
# ten = top10.iloc[9]
# print(ten)
#
# data['crim'] = np.where(data['crim']>=ten,ten,data['crim'])
#
# result = data[data['age']>=80]
#
# print(result['age'].describe())
#
#
# print(round(result['crim'].mean(),2))

#작업형 1 - 2
# import pandas as pd
# import numpy as np
# pd.set_option('mode.chained_assignment',  None)
# data = pd.read_csv('dataset/P210202.csv')
#
# print(len(data))
#
# train = data.head(int(20640*0.8))
# result1 = train['total_bedrooms'].std()
# train['total_bedrooms'] = train['total_bedrooms'].fillna(train['total_bedrooms'].median())
# result2 = train['total_bedrooms'].std()
#
# print(round(abs(result1-result2),2))

#작업형 1 - 3
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210203.csv')
#
#
# mini = data['charges'].mean()-(data['charges'].std()*1.5)
# maxi = data['charges'].mean()+(data['charges'].std()*1.5)
#
# result = data[data['charges']<= mini]
# result = data[data['charges']>= maxi]
#
# print(int(result['charges'].sum()))

#작업형 1 - 4
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210301.csv')
#
# cut = int(len(data)*0.7)
#
# result = data.iloc[:cut,:]
#
# result = result.dropna()
#
# answer = np.percentile(result['housing_median_age'],25)
# print(int(answer))

#작업형 1 - 5
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210302.csv')
# result = data.isna().sum()/len(data)
# result = pd.DataFrame(result)
# result = result.sort_values(by=0, ascending = False)
# print(result.index[0])

#작업형 1 - 6
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P210303.csv')
#
# data = data[['country', 'year', 'new_sp']]
# data = data.dropna()
# data = data[data['year']==2000]
# country_mean = round(data["new_sp"].mean(),2)
# print(country_mean)
# result = data[data["new_sp"]>=country_mean]
# print(len(result))

#작업형 1 - 7
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P220401.csv')
#
# y1 = np.percentile(data['y'],25)
# y2 = np.percentile(data['y'],75)
# result = int(y2-y1)
# print(result)

#작업형 1 - 8
# import pandas as pd
# import numpy as np
# #좋아요 놀랐어요 긍정 으로 치고 비율 계산해서 0.4 ~ 0.5
# data = pd.read_csv('dataset/P220402.csv')
#
# data['temp'] = (data['num_loves']+data['num_wows'])/data['num_reactions']
#
# result = data[data['temp']>0.4]
# result = result[result['temp']<0.5]
# result = result[result['status_type']=='video']
# answer = len(result)
# print(answer)

#작업형 1 - 9
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P220501.csv')
# print(data.columns)
# data = data[data['종량제봉투용도']=="음식물쓰레기"]
# data = data[data['종량제봉투종류']=="규격봉투"]
# data = data[data['2L가격']>0]
# answer = int(round(data['2L가격'].mean(),0))
# print(answer)