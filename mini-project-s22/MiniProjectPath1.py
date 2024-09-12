import pandas
import numpy
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets
from sklearn.metrics import  r2_score
import math
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
dataset = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
highTemp = pandas.to_numeric(dataset['High Temp'])
lowTemp = pandas.to_numeric(dataset['Low Temp'])
precipitation = pandas.to_numeric(dataset['Precipitation'])
rain = pandas.to_numeric(dataset['Precipitation'])
rain = [int(precipitation !=0) for precipitation in rain]
bTraffic = pandas.to_numeric(dataset['Brooklyn Bridge'].replace(',','', regex=True))
mTraffic = pandas.to_numeric(dataset['Manhattan Bridge'].replace(',','', regex=True))
wTraffic  = pandas.to_numeric(dataset['Williamsburg Bridge'].replace(',','', regex=True))
qTraffic  = pandas.to_numeric(dataset['Queensboro Bridge'].replace(',','', regex=True))
totalTraffic = pandas.to_numeric(dataset['Total'].replace(',','', regex=True))

#print(dataset_1.to_string()) #This line will print out your data


"""The use of the code provided is optional, feel free to add your own code to read the dataset. The use (or lack of use) of this code is optional and will not affect your grade."""


def norm_histogram(hist):
    """
    takes a histogram of counts and creates a histogram of probabilities
    :param hist: a numpy ndarray object
    :return: list
    """
    # Create list
    list = []
    
    # Calculate and add probability to list
    for n in range(len(hist)):
        probability = hist[n] / sum(hist)
        list.append(probability)

    # Return list
    return list

# Descriptive Statistics

sampleSize = len(highTemp)
# Calculate mean and standard deviation
meanHigh = numpy.mean(highTemp)
meanLow = numpy.mean(lowTemp)
meanPrecipitation = numpy.mean(precipitation)
mean_b = numpy.mean(bTraffic)
mean_m = numpy.mean(mTraffic)
mean_w = numpy.mean(wTraffic)
mean_q = numpy.mean(qTraffic)
mean_total = numpy.mean(totalTraffic)

stdHigh = numpy.std(highTemp)
stdLow = numpy.std(lowTemp)
stdPrecipitation = numpy.std(precipitation)
std_b = numpy.std(bTraffic)
std_m = numpy.std(mTraffic)
std_w = numpy.std(wTraffic)
std_q = numpy.std(qTraffic)
std_total = numpy.std(totalTraffic)

# Show histogram and normal histogram of total traffic
days = numpy.arange(0,sampleSize, 1)
norm_total = norm_histogram(totalTraffic)
plt.figure(1)
plt.hist(totalTraffic, bins = int(math.sqrt(sampleSize)))
plt.title('Histogram of Total Traffic')
plt.xlabel('Traffic')
plt.ylabel('Frequency')

plt.figure(2)
plt.scatter(days, totalTraffic, c = 'black', label = 'Total Traffic')
plt.title('Scatter plot of Total Traffic')
plt.xlabel('Day')
plt.ylabel('Traffic')
plt.show()

# Problem 1

# Convert traffics to proportion of total traffic on that specific day
bProportion = pandas.to_numeric(dataset['Brooklyn Bridge'].replace(',','', regex=True))
mProportion = pandas.to_numeric(dataset['Manhattan Bridge'].replace(',','', regex=True))
wProportion  = pandas.to_numeric(dataset['Williamsburg Bridge'].replace(',','', regex=True))
qProportion = pandas.to_numeric(dataset['Queensboro Bridge'].replace(',','', regex=True))

for i in range(sampleSize):
    bProportion[i] = bProportion[i] / totalTraffic[i]
    mProportion[i] = mProportion[i] / totalTraffic[i]
    wProportion[i] = wProportion[i] / totalTraffic[i]
    qProportion[i] = qProportion[i] / totalTraffic[i]
# Get average proportions
meanbProportion = numpy.mean(bProportion)
meanmProportion = numpy.mean(mProportion)
meanwProportion = numpy.mean(wProportion)
meanqProportion = numpy.mean(qProportion)
# Get scaling factor (used to estimate overall traffic) based on average proportions
bscalingfactor = 1 / meanbProportion
mscalingfactor = 1 / meanmProportion
wscalingfactor = 1 / meanwProportion
qscalingfactor = 1 / meanqProportion

# Get estimate of overall traffic per day using scaling factor
bEstimate = bTraffic * bscalingfactor
mEstimate = mTraffic * mscalingfactor
wEstimate = wTraffic * wscalingfactor
qEstimate = qTraffic * qscalingfactor
# Get regression models from estimates
bmodel = numpy.poly1d(numpy.polyfit(days, bEstimate,3))
mmodel = numpy.poly1d(numpy.polyfit(days, mEstimate,3))
wmodel = numpy.poly1d(numpy.polyfit(days, wEstimate,3))
qmodel = numpy.poly1d(numpy.polyfit(days, qEstimate,3))

# Get R-squared values of models
bRsquared = r2_score(totalTraffic, bmodel(days))
mRsquared = r2_score(totalTraffic, mmodel(days))
wRsquared = r2_score(totalTraffic, wmodel(days))
qRsquared = r2_score(totalTraffic, qmodel(days))

print(f"Brooklyn model R-squared value = {bRsquared}")
print(f"Manhattan model R-squared value = {mRsquared}")
print(f"Williamsburg model R-squared value = {wRsquared}")
print(f"Queensboro model R-squared value = {qRsquared}")

#Smallest R-squared value is of the model that is the least accurate

# Create scatter plot
""" plt.figure(3)
plt.subplot(2,2,1)
plt.scatter(days, bTraffic, c = 'red', label = 'Total Traffic')
plt.scatter(days, totalTraffic, c = 'black', label = 'Total Traffic')
plt.subplot(2,2,2)
plt.scatter(days, mTraffic, c = 'orange', label = 'Total Traffic')
plt.scatter(days, totalTraffic, c = 'black', label = 'Total Traffic')
plt.subplot(2,2,3)
plt.scatter(days, wTraffic, c = 'green', label = 'Total Traffic')
plt.scatter(days, totalTraffic, c = 'black', label = 'Total Traffic')
plt.subplot(2,2,4)
plt.scatter(days, qTraffic, c = 'blue', label = 'Total Traffic')
plt.scatter(days, totalTraffic, c = 'black', label = 'Total Traffic') """
plt.figure(3)
plt.plot(days, bmodel(days), c = 'red', label = 'Brooklyn Estimate Model of Total')
plt.plot(days, mmodel(days), c = 'orange', label = 'Manhattan Estimate Model of Total')
plt.plot(days, wmodel(days), c = 'green', label = 'Williamsburg Estimate Model of Total')
plt.plot(days, qmodel(days), c = 'blue', label = 'Queensboro Estimate Model of Total')
plt.scatter(days, totalTraffic, c = 'black', label = 'Total Traffic')
plt.legend(fontsize=10)
plt.xlabel('Days')
plt.ylabel('Traffic')
plt.title('Plot of Total Traffic and Estimates')
plt.show()

# Problem 2
X = numpy.array([highTemp, lowTemp, precipitation]).T
y = numpy.array(totalTraffic).T

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# column names
col = (dataset[["High Temp", "Low Temp", "Precipitation"]]).columns
# coefficients of features
y_pred = regressor.predict(X_test)
print("Problem 2 model r-squared score: ", regressor.score(X_test, y_test))

# Problem 3
# Classification raining or not -> Confusion matrix
X = numpy.array(totalTraffic)
X = X.reshape(-1,1)
Y = numpy.array(rain).T
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

model = GaussianNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Problem 3 model r-squared score: ", model.score(X_test, y_test))
print("Raining or Not (Test set)")
print(y_test)
print("Raining or Not (Prediction)")
print(y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)