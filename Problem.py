

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
import pandas as pd
import numpy as np

data = pd.read_csv('dataset/P210203.csv')


mini = data['charges'].mean()-(data['charges'].std()*1.5)
maxi = data['charges'].mean()+(data['charges'].std()*1.5)

result = data[data['charges']<= mini]
result = data[data['charges']>= maxi]

print(int(result['charges'].sum()))