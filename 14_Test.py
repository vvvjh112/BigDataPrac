import pandas as pd
import numpy as np
from scipy.stats import *




#검정 - 모집단의 평균이 표본평균과 차이가 있는지 검정하는 방법
#모집단의 분산을 알고있는 경우 Z검정 / 모를경우 T검정
#두 검정 모두 모집단이 정규성을 따라야 한다.

#가설설정 -> 유의수준 설정 -> 정규성 검정 -> 검정통계량 및 p-값 산출, 귀무가설 기각 여부 순서로 진행


#Z검정 / 귀무가설 모평균은 26 (모평균 = 표본평균) 표준편차 5
print("Z검정")
x = np.array([25,27,31,23,24,30,26])
z = (np.mean(x) - 26) / 5 * np.sqrt(7)
print(z)
p_value = (1-norm.cdf(z))*2
#z의 위치까지 누적된 분포 함수의 값을 계산하고, p값을 얻기 위해 전체 면적 1 에서 차감 / 양측 검정이므로 2곱함
print(p_value)
#값이 약 0.76로 유의수준보다 크기 때문에 귀무가설 채택


#T검정
#T검정은 독립변수가 범주형이고, 종속변수가 수치형일 때 두 집단의 평균을 비교하는 검정 방법
#표본이 정규성, 등분산성, 독립성등을 만족할 경우 적용
print("T검정")

#T검정 단일표본
#T검정을 하기전에 정규분포를 따른다는 정규성 가정을 확인해야한다.
#샤피로윌크 검정을 사용하며 p값과 유의수준과 비교해서 확인 / 귀무가설 : 데이터가 정규분포를 따름

#신제품 7개의 높이가 df 유의수준 0.05 / 평균 11cm인지 아닌지 양측검정
from scipy.stats import *

df = pd.DataFrame({'height' : [12,14,16,19,11,17,13]})
print(shapiro(df['height']))
#(statistic=0.9641615748405457, pvalue=0.8535423278808594)
print(ttest_1samp(df['height'],popmean = 11))
#p값이 유의수준보다 낮으므로 대립가설인 평균은 11이 아니다 채택


#정규성을 만족하지 않는 경우
#wilcoxon함수를 사용하여 검정 수행

#고양이들의 평균 몸무게가 2.1kg인지 아닌지에 대해서 양측검정
import pandas as pd
from scipy.stats import *

df = pd.read_csv('dataset/cats.csv')
result = shapiro(df['Bwt'])
print(result)

result = wilcoxon(df['Bwt']-2.1, alternative='two-sided')
#기준값을 뺌
print(result)
#대립가설채택


#쌍체표본 T검정
#한 집단에서 처치를 받기 전과 후의 차이를 알아보기 위해 사용하는 검정 방법
#ttest_rel(x,y,alternative)
#종속변수는 연속형이어야 하며, 정규성 가정을 만족해야 한다.

import pandas as pd
from scipy.stats import *

data = pd.DataFrame({'before':[5,3,8,4,3,2,1],'after':[8,6,6,5,8,7,3]})

result = ttest_rel(data['before'],data['after'],alternative = 'less')

print(result)
#대립가설 채택

#독립표본 T검정
#데이터가 서로 다른 모집단에서 추출된 경우 사용할 수 있는 검정
#두 집단의 평균 차이를 검정 / 정규성, 등분산성 가정이 만족되는지 확인
#독립변수는 범주형, 종속변수는 연속형이어야 한다.
#등분산 검정 -> levene(sample1, sample2, center) / center -> median or mean
#ttest_ind(sample1, sample2, alternative, equal_var)

import pandas as pd
from scipy.stats import *
#성별에 따른 몸무게 변화의 차이가 있다 없다
cats = pd.read_csv('dataset/cats.csv')
group1 = cats[cats['Sex']=="F"]['Bwt']
group2 = cats[cats['Sex']=="M"]['Bwt']

result = levene(group1,group2)
print(result) #등분산성 만족x
result = ttest_ind(group1, group2, equal_var = False)
print(result)
# 대립가설 채택


print("F검정")
#두 표본의 분산에 대한 차이가 통계적으로 유의한가를 판별하는 검정
#f.cdf(x, dfn, dfd) / F검정 통계량, F분포의 분자의 자유도, F분포의 본모의 자유도

import numpy as np
df1 = np.array([1,2,3,4,6])
df2 = np.array([4,5,6,7,8])

print(np.var(df1), np.var(df2))

def f_test(x, y):
    if np.var(x, ddof=1) < np.var(y,ddof=1):
        x,y = y,x
    f_value = np.var(x, ddof=1) / np.var(y,ddof=1)
    x_dof = x.size -1
    y_dof = y.size -1
    p_value = (1-f.cdf(f_value, x_dof, y_dof)) * 2
    return f_value, p_value

result = f_test(df1, df2)
print(result)

#카이제곱검정
print("카이제곱 검정")
#범주형 자료간의 차이를 보여주는 분석 방법 / 관찰된 빈도가 기대되는 빈도와 유의하게 다른지 검정
# 교차분석은 적합도검정, 독립성 검정, 동질성검정 3가지로 분류

#적합도검정 (자유도 = 범주의 수 - 1)
# 표본 집단의 분포가 주어진 특정분포를 따르고 있는지를 검정
import numpy as np
from scipy.stats import *
#남학생 90명 여학생 160명이 있는데, 비율이 45%와 55%인지 확인
num = np.array([90,160])
expected = np.array([0.45,0.55])*np.sum(num) # 250명중 45%가 남자 55%가 여자이기때문

result = chisquare(num, f_exp=expected)
print(result)
#대립가설채택


#독립성 검정
# 변수가 두 개 이상의 범주로 분할되어 있을 때 사용되며, 각 범주가 서로 독립적인지 서로 연관성 있는지 검정
#자유도 = (범주1의 수 -1) * (범주2의 수 -1)
import pandas as pd
from scipy.stats import *
