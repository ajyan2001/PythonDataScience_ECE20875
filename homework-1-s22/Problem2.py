#!/usr/bin/python3
import math
n = 21
# Your code should be below this line

# get day of week
day = n % 7
# if n is invalid
if n < 1 or n > 31:
    print('Not valid')
# if day is 1-5, weekday
elif day >= 0 and day <= 5:
    print('Weekday')
# if day is 6-7, weekend
else:
    print('Weekend')