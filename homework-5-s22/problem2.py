import math as m
from pydoc import doc
import numpy as np
import scipy.stats as stats

# import or paste dataset here
scores = [3, -3, 3, 12, 15, -16, 17, 19, 23, -24, 32]
dfree = len(scores) - 1
# code for question 1
print('Problem 1 Answers:')
# code below this line
mean = np.mean(scores)
print(f'    The sample mean is {mean}')

se = np.std(scores, ddof = dfree) / np.sqrt(len(scores))
print(f'    The standard deviation is {se}')

tval1 = stats.t.ppf(1 - (1 - 0.90) / 2,dfree)
print(f'    The t-value of 90% confidence interval is {tval1}')

min1 = mean - (tval1 * se)
max1 = mean + (tval1 * se)
print(f'    The 90% confidence interval is ({min1},{max1})')

# code for question 2
print('Problem 2 Answers:')
# code below this line
tval2 = stats.t.ppf(1 - (1 - 0.95) / 2, dfree)
print(f'    The t-value of 95% confidence interval is {tval2}')

min2 = mean - (tval2 * se)
max2 = mean + (tval2 * se)
print(f'    The 95% confidence interval is ({min2},{max2})')
# code for question 3
print('Problem 3 Answers:')
# code below this line
se2 = 15.836 / (len(scores) ** 0.5)
print(f'    The standard error is {se2}')

zscore = stats.norm.ppf(1 - (1 - 0.95) / 2)
print(f'    The z-score of the 95% confidence interval is {zscore}')
min3 = mean - (zscore * se2)
max3 = mean + (zscore * se2)
print(f'    The 95% confidence interval is ({min3},{max3}')


# code for question 4
print('Problem 4 Answers:')
# code below this line
# tval = mean / se
c = 0.95
tvalue = stats.t.ppf(1 - (1 - c) / 2, dfree)
min = mean - (tvalue * se)

while min < 0:
    c -= 0.01
    tvalue = stats.t.ppf(1 - (1 - c) / 2, dfree)
    min = mean - (tvalue * se)

print(f'The confidence level required for the lower endpoint to be 0 is {c * 100}%')