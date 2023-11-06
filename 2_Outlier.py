import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

prac = pd.DataFrame({'number':[1,10000,1,1,1,1,1,1,1,1,1],'type':['a','b','c','d','e','f','g','h','i','j','k']})

# ESD를 이용한 이상값 검출(3표준편차 이내의 값들이면 True 반환)
def esd(x):
    return abs((x-x.mean())/x.std())<3

# print(train[esd(train['hour_bef_temperature'])])
print(prac[esd(prac['number'])])

#박스플롯이용
import matplotlib.pyplot as plt

df = pd.DataFrame({"score":[65,60,70,75,200], 'name':['A','B','C','D','E']})

box = plt.boxplot(df['score'])

print(box.keys())

mini = box['whiskers'][0].get_ydata()[1]
maxi = box['whiskers'][1].get_ydata()[1]
q1 = box['boxes'][0].get_ydata()[1]
q2 = box['medians'][0].get_ydata()[0]
q3 = box['boxes'][0].get_ydata()[2]
outliers = []
for i in df['score']:
    if (i < mini) or (i > maxi):
        outliers.append(i)

print("최솟값 : {}, 최댓값 : {}, 1사분위 : {}, 2사분위(중위) : {}, 3사분위 : {}".format(mini, maxi, q1, q2, q3))
print("이상값 :",outliers)

#IQR을 이용해서 검출
from scipy.stats import iqr

min_s = df['score'].median()-iqr(df['score'])
max_s = df['score'].median()+iqr(df['score'])

print(df[(df['score']>=min_s) & (df['score']<=max_s)])