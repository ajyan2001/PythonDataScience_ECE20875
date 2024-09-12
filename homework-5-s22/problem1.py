import numpy as np
import math as m
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import t

# import or paste dataset here
eng0 = open('engagement_0.txt')
data0 = eng0.readlines()
eng0.close()

eng1 = open('engagement_1.txt')
data1 = eng1.readlines()
eng1.close()

data0 = [float(x) for x in data0]
data1 = [float(x) for x in data1]

# code for question 2
print('Problem 2 Answers:')
# code below this line
# sample size
sampleSize = len(data1)
print(f'  Sample size = {sampleSize}')
# sample mean
sampleMean = np.mean(data1)
print(f'  Sample mean = {sampleMean}')
# standard error
sd = np.std(data1, ddof = 1)
print(f'  Standard deviation = {sd}')
# standard score
zScore = (sampleMean - 0.75) / sd
print(f'  Z-score = {zScore}')
# standard p-value
p = 2 * norm.cdf(-abs(zScore))
print(f'  P-value = {p}')


# code for question 3
print('Problem 3 Answers:')
# code below this line
# Required z score
z = norm.cdf(0.05)
# Require standard deviation
newSd = abs((sampleMean - 0.75) / z)
# population Variance
pVar = sd * (sampleSize) ** 0.5
n = (pVar / newSd) ** 2
print(f'  Standard Deviation required for 0.05 significance: {newSd}')
print(f'  Corresponding sample size: {n}')

# code for question 5
print('Problem 5 Answers:')
# code below this line
# Sample Sizes
size0 = len(data0)
size1 = len(data1)
print(f'Sample size of engagement0: {size0}')
print(f'Sample size of engagement1: {size1}')
# Sample means
mean0 = np.mean(data0)
mean1 = np.mean(data1)
print(f'  Sample mean of engagement0: {mean0}')
print(f'  Sample mean of engagement1: {mean1}')
# standard error
sd0 = np.std(data0, ddof = 1)
sd1 = np.std(data1, ddof = 1)
print(f'  Standard deviation of engagement0: {sd0}')
print(f'  Standard deviation of engagement1: {sd1}')
# Z score
zscore = (mean0 - mean1) / (mean0 ** 2 / size0 + mean1 ** 2 / size1) ** 0.5
print(f'  Z-score = {zscore}')
# p value
pvalue = 2 * norm.cdf(-abs(zscore))
print(f'  P-value = {pvalue}')
