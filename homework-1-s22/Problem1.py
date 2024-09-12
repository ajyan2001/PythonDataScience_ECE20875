#!/usr/bin/python3
import math
number = 100
# Your code should be below this line
# Fibonacci conditions
fibCon1 = math.sqrt(5 * number ** 2 + 4) * 10 % 10
fibCon2 = math.sqrt(5 * number ** 2 - 4) * 10 % 10
#check condition is true
if (fibCon1 == 0 or fibCon2 == 0) and number % 2 == 0 and number > 0:
    print('Yes')
#if condition is false
else:
    print('No')