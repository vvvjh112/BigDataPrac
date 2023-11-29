from scipy.stats import ttest_rel

# 데이터
before = [80,75,90,60,85]
after = [85,70,92,65,88]

# t-검정 수행
t_statistic, p_value = ttest_rel(before, after,alternative = 'greater')

# 결과 출력
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")

# 유의수준(alpha)과 비교하여 귀무가설 기각 여부 판단
alpha = 0.05
if p_value < alpha:
    print("귀무가설 기각: 평균 매출은 통계적으로 유의미하게 차이가 있습니다.")
else:
    print("귀무가설 채택: 평균 매출은 통계적으로 유의미한 차이가 없습니다.")