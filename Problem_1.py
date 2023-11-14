

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
# import pandas as np
#
#
# data = pd.read_csv('dataset/P220403.csv')
# data = data[data['country'] == "United Kingdom"]
# # print(data['date_added'].head())
# data['date_added'] = pd.to_datetime(data['date_added'], format = "%B %d, %Y")
# data = data[data['date_added'].dt.year == 2018]
# data = data[data['date_added'].dt.month == 1]
#
# answer = len(data)
#
# print(answer)


#작업형 1 - 10
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


#작업형 1 - 11
# import pandas as pd
# import numpy as np
#
# data = pd.read_csv('dataset/P220502.csv')
# #무게 / 키(m)^2
# #18.5 < bmi 저체중 18.5<= bmi <23 정상 23<= bmi < 25 위험 25<= bmi 비만
# #정상체중범위 구간의 인원과 위험체중 범위의 인원 ㅊ ㅏ이 절댓값
#
# def bmi(bm):
#
#     if bm < 18.5:
#         return "저체중"
#     elif bm < 23:
#         return "정상"
#     elif bm <25:
#         return "위험"
#     else:
#         return "비만"
#
# data['bmi'] = data['Weight'] / (pow((data['Height']/100),2) )
# data['tmp'] = data['bmi'].apply(bmi)
# print(data['bmi'].head())
# normal = len(data[data['tmp'] == "정상"])
# warning = len(data[data['tmp'] == "위험"])
#
# answer = int(abs(normal-warning))
#
# print(answer)

#작업형 1 - 12
# import pandas as pd
#
# data = pd.read_csv('dataset/P220503.csv')
# print(data.columns)
# data['tmp'] = data['전입학생수(계)'] - data['전출학생수(계)']
#
# data = data.sort_values('tmp',ascending = False)
# answer = data['전체학생수(계)'].iloc[0]
# print(answer)


#작업형 1 - 13
# import pandas as pd
# import datetime
#
# data = pd.read_csv('dataset/P230601.csv')
#
# data['신고일시'] = pd.to_datetime(data['신고일시'])
# data['출동일시'] = pd.to_datetime(data['출동일시'])
#
# data['time'] = (data['출동일시'] - data['신고일시']).dt.total_seconds()
#
# data = data.groupby([data['출동소방서'],data['신고일시'].dt.year,data['신고일시'].dt.month]).mean('time')
#
# data = data.sort_values('time',ascending = False)
# result = data['time'].head(1)
# answer = int(round(result.iloc[0]/60,0))
#
# print(answer)

#작업형 1 - 14
# import pandas as pd
#
# data = pd.read_csv('dataset/P230602.csv')
# print(data.columns)
# data['sum'] = data['student_1']+data['student_2']+data['student_3']+data['student_4']+data['student_5']+data['student_6']
# data['tmp'] = data['sum']/data['teacher']
#
# data = data.sort_values('tmp',ascending = False)
#
# result = data.head(1)
# answer = result['teacher']
# answer = answer.iloc[0]
# print(answer)

#작업형 1 - 15
# import pandas as pd
# import numpy as np
# import datetime
#
# data = pd.read_csv('dataset/P230603.csv')
# data1 = pd.read_csv('dataset/P230603.csv')
# data['년월'] = pd.to_datetime(data['년월'])
# data['년월'] = data['년월'].apply(lambda x: datetime.datetime.strftime(x,"%y"))
# data['범죄총계'] = data['강력범'] + data['절도범'] + data['폭력범'] + data['지능범'] + data['풍속범'] + data['기타형사범']
# result = data.groupby("년월").sum("범죄총계")
# result = result.sort_values("범죄총계",ascending = False)
# df = result.head(1)
# answer = int(result["범죄총계"].iloc[0]/12)
#
# print(answer)