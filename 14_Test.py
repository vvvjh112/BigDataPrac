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
