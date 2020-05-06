from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sns
import random
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stt
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso
from util import Prediction
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from keras.models import Sequential
from keras.layers import Dense, Activation


###########################################  Data Visualization  #############################################

# read the data set( from world happiness report)
data19 = pd.read_csv('/Users/apple/Desktop/ucla winter/425/project/2019.csv')
data19.head()

# distribution of happiness scores
fig = plt.figure(figsize=(7, 5))
sns.set()
sns.distplot(data19['Score'], bins=15);
plt.title('Histogram of Happiness Scores')
plt.show()

# Variables summary
summary = data19.describe().T
print(summary)

# scatter plot through each variables
fig = plt.figure(figsize=(14, 12))
sns.set(style="white", font_scale=1);
sns.pairplot(data19[['Score', 'Country or region', 'Social support', 'GDP per capita', 'Healthy life expectancy',
                     'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']])
plt.show()

# correlation plot through each variables
fig = plt.figure(figsize=(19, 12))
sns.set(style="white", font_scale=1.5)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(data19.dropna()[
                ['Score', 'Country or region', 'Social support', 'GDP per capita', 'Healthy life expectancy',
                 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']].corr(), fmt='.2f',
            annot=True, \
            xticklabels=False, linewidth=2, cbar_kws={"shrink": .5}, cmap=cmap);
plt.title('Correlation Plot', fontsize=40)
plt.show()


plt.rcParams['figure.figsize'] = (20, 15)
d = data19.loc[lambda data19: data19['Region'] == 'Western Europe']
sns.heatmap(d.corr(), cmap = 'Wistia', annot = True)
plt.title('Western Europe Condition',fontsize=20)
plt.show()

plt.rcParams['figure.figsize'] = (20, 15)
d = data19.loc[lambda data19: data19['Region'] == 'Eastern Asia']
sns.heatmap(d.corr(), cmap = 'Greys', annot = True)
plt.title('Eastern Asia Condition',fontsize=20)
plt.show()

plt.rcParams['figure.figsize'] = (20, 15)
d = data19.loc[lambda data19: data19['Region'] == 'North America']
sns.heatmap(d.corr(), cmap = 'pink', annot = True)
plt.title('North America Condition',fontsize=20)
plt.show()

plt.rcParams['figure.figsize'] = (20, 15)
d = data19.loc[lambda data19: data19['Region'] == 'Middle East and Northern Africa']
sns.heatmap(d.corr(), cmap = 'rainbow', annot = True)
plt.title('Middle East and Northern Africa Condition',fontsize=20)
plt.show()

plt.rcParams['figure.figsize'] = (20, 15)
d = data19.loc[lambda data19: data19['Region'] == 'Sub-Saharan Africa']
sns.heatmap(d.corr(), cmap = 'Blues', annot = True)
plt.title('Sub-Saharan Africa Condition',fontsize=20)
plt.show()


#########################################  Data Processing  #############################################
# Choose variables whose correlation coefficients are bigger than 5
x = data19[['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices']]
y = data19[['Score']]
x = np.array(x)
y = np.array(y)

# Data splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Data normalization
def rescaleNormalization(dataArray):
    min = dataArray.min()
    denom = dataArray.max() - min
    newValues = []
    for x in dataArray:
        newX = (x - min) / denom
        newValues.append(newX)
    return newValues

x_train = rescaleNormalization(np.array(x_train))
x_test = rescaleNormalization(np.array(x_test))
y_train = rescaleNormalization(np.array(y_train))
y_test = rescaleNormalization(np.array(y_test))
#########################################  Data Processing  #############################################


###########################################  Linear regression  ##########################################

# training and testing using sklearn
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)
print("score Scikit learn: ", linear_regression.score(x_train, y_train))
coeffs = linear_regression.coef_  # coefficients
intercept = linear_regression.intercept_  # bias
bHat = np.hstack((np.array([intercept]), coeffs))
print(bHat)

xx_test = np.ones((63, 5))
xx_test[:, 1:5] = np.array(x_test)[:, 0:4]
linear_prediction1= np.dot(xx_test,np.transpose(bHat))
RMSE_linear_regression1=np.sqrt(mean_squared_error(linear_prediction1, y_test))
print(RMSE_linear_regression1)


# training and testing using self-developed model
xx_train = np.ones((93, 5))
xx_train[:, 1:5] = np.array(x_train)[:, 0:4]

def gradientDescent(X, y, theta, alpha, numIterations):
    m = len(y)
    arrCost = [];
    transposedX = np.transpose(X)  #
    for interation in range(0, numIterations):
        residualError = (np.sum(theta * X, axis=1)) - np.transpose(y)
        gradient = np.sum(residualError * transposedX, axis=1) / m
        change = [alpha * x for x in gradient]
        theta = np.subtract(theta, change)
        atmp = (1 / (2 * m)) * np.sum(residualError ** 2)
        arrCost.append(atmp)
    return theta

ALPHA = 0.005
MAX_ITER = 1000
theta = np.zeros(5)
theta_self = gradientDescent(np.array(xx_train), np.array(y_train), theta, ALPHA, MAX_ITER)
print(theta_self)

linear_prediction2= np.dot(xx_test,np.transpose(theta_self))
RMSE_linear_regression2=np.sqrt(mean_squared_error(linear_prediction2, y_test))
print(RMSE_linear_regression2)


rmse_LR = []
models_LR = [Ridge(), Lasso()]
model_names_LR = ['Ridge', 'Lasso']
for i in range(len(models_LR)):
    clf = models_LR[i]
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    rmse_LR.append(np.sqrt(mean_squared_error(pred, y_test)))
d_LR = {'Modelling Algorithm': model_names_LR, 'RMSE': rmse_LR}
RMSE_LR_df = pd.DataFrame(d_LR)
print(RMSE_LR_df)


fig2, axes = plt.subplots(1, 2)
axes[0].scatter(y_test, linear_prediction1,s=50)
axes[0].set_title(" linear regression using sklearn")
axes[0].set_xlabel("true values")
axes[0].set_ylabel("predicted values")
axes[1].scatter(y_test, linear_prediction2)
axes[1].set_title("linear regression using self-developed method")
axes[1].set_xlabel("true values")
axes[1].set_ylabel("predicted values")
fig2.set_size_inches(10, 7)
plt.show()

###########################################  Polynomial regression  ###########################################
poly_reg = PolynomialFeatures(degree = 3)
x_poly_train = poly_reg.fit_transform(xx_train)
x_poly_test  = poly_reg.fit_transform(xx_test)
poly_reg.fit(x_poly_train, y_train)

lin_reg = LinearRegression()
lin_reg.fit(x_poly_test, y_test)

polynomial_pred = lin_reg.predict(x_poly_test)
RMSE_polynomial=np.sqrt(mean_squared_error(polynomial_pred, y_test))
print(RMSE_polynomial)


#################################################  SVR  #####################################################
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,param_grid={"C": [1e0, 1e1, 1e2, 1e3],"gamma": np.logspace(-2, 2, 5)})
svr.fit(xx_train[:93], np.ravel(y_train[:93]))
y_svr = svr.predict(xx_test)
RMSE_SVR = np.sqrt(mean_squared_error(y_svr, np.ravel(y_test)))
print(RMSE_SVR)


###########################################  Random Forest Regression  ######################################
random_regression = RandomForestRegressor(n_estimators = 15, random_state = 63)
random_regression.fit(x_train, y_train)
y_pred = random_regression.predict(x_test)
RSME_forest = np.sqrt(mean_squared_error(y_pred, y_test))
print(RSME_forest)


###########################################  Natural Network  ###############################################
model_NN = Sequential()
model_NN.add(Dense(1000, input_dim=4, activation='tanh'))
model_NN.add(Dense(1000, activation='tanh'))
model_NN.add(Dense(1,activation='tanh'))
model_NN.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])
model_NN.fit(np.array(x_train), np.array(y_train), epochs=10, batch_size=5)
score = model_NN.evaluate(np.array(x_test), np.array(y_test))

pred = model_NN.predict(np.array(x_test))
RMSE_NN = np.sqrt(mean_squared_error(pred, y_test))
print(RMSE_NN)

###########################################  Other Regression  ######################################
models_EM = [AdaBoostRegressor(), BaggingRegressor(), KNeighborsRegressor()]
model_names_EM = ['AdaBoostRegressor', 'BaggingRegressor', 'KNeighborsRegressor']
rmse_EM = []
for i in range(len(models_EM)):
    clf = models_EM[i]
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    rmse_EM.append(np.sqrt(mean_squared_error(pred, y_test)))
d_EM = {'Modelling Algorithm': model_names_EM, 'RMSE': rmse_EM}
rmse_EM_df = pd.DataFrame(d_EM)
print(rmse_EM_df)



fig3, axes = plt.subplots(2, 2)
axes[0,0].scatter(y_test, polynomial_pred,s=50)
axes[0,0].set_title("Polynomial regression")
axes[0,0].set_xlabel("true values")
axes[0,0].set_ylabel("predicted values")
axes[0,1].scatter(y_test, y_svr)
axes[0,1].set_title("Support vector regression")
axes[0,1].set_xlabel("true values")
axes[0,1].set_ylabel("predicted values")
axes[1,0].scatter(y_test, y_pred)
axes[1,0].set_title("Random forest regression")
axes[1,0].set_xlabel("true values")
axes[1,0].set_ylabel("predicted values")
axes[1,1].scatter(y_test, pred)
axes[1,1].set_title("Neural Network")
axes[1,1].set_xlabel("true values")
axes[1,1].set_ylabel("predicted values")

fig3.set_size_inches(10, 10)
plt.show()

