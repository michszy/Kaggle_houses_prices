# =============================================================================
# meilleurs correlation avec le SalePrice:
#Overall Qual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearRemodAdd > 50%
#
#Columns #'Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
#       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
#       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
#       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
#       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
#       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
#       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
#       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
#       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
#       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
#       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
#       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
#       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
#       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
#       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
#       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
#       'SaleCondition', 'SalePrice'
#sns.distplot(train['SalePrice'])
#print(train['SalePrice'].describe())

#Id               1460 non-null int64
#MSSubClass       1460 non-null int64
#MSZoning         1460 non-null object
#LotFrontage      1201 non-null float64
#LotArea          1460 non-null int64
#Street           1460 non-null object
#Alley            91 non-null object
#LotShape         1460 non-null object
#LandContour      1460 non-null object
#Utilities        1460 non-null object
#LotConfig        1460 non-null object
#LandSlope        1460 non-null object
#Neighborhood     1460 non-null object
#Condition1       1460 non-null object
#Condition2       1460 non-null object
#BldgType         1460 non-null object
#HouseStyle       1460 non-null object
#OverallQual      1460 non-null int64
#OverallCond      1460 non-null int64
#YearBuilt        1460 non-null int64
#YearRemodAdd     1460 non-null int64
#RoofStyle        1460 non-null object
#RoofMatl         1460 non-null object
#Exterior1st      1460 non-null object
#Exterior2nd      1460 non-null object
#MasVnrType       1452 non-null object
#MasVnrArea       1452 non-null float64
#ExterQual        1460 non-null object
#ExterCond        1460 non-null object
#Foundation       1460 non-null object
#BsmtQual         1423 non-null object
#BsmtCond         1423 non-null object
#BsmtExposure     1422 non-null object
#BsmtFinType1     1423 non-null object
#BsmtFinSF1       1460 non-null int64
#BsmtFinType2     1422 non-null object
#BsmtFinSF2       1460 non-null int64
#BsmtUnfSF        1460 non-null int64
#TotalBsmtSF      1460 non-null int64
#Heating          1460 non-null object
#HeatingQC        1460 non-null object
#CentralAir       1460 non-null object
#Electrical       1459 non-null object
#1stFlrSF         1460 non-null int64
#2ndFlrSF         1460 non-null int64
#LowQualFinSF     1460 non-null int64
#GrLivArea        1460 non-null int64
#BsmtFullBath     1460 non-null int64
#BsmtHalfBath     1460 non-null int64
#FullBath         1460 non-null int64
#HalfBath         1460 non-null int64
#BedroomAbvGr     1460 non-null int64
#KitchenAbvGr     1460 non-null int64
#KitchenQual      1460 non-null object
#TotRmsAbvGrd     1460 non-null int64
#Functional       1460 non-null object
#Fireplaces       1460 non-null int64
#FireplaceQu      770 non-null object
#GarageType       1379 non-null object
#GarageYrBlt      1379 non-null float64
#GarageFinish     1379 non-null object
#GarageCars       1460 non-null int64
#GarageArea       1460 non-null int64
#GarageQual       1379 non-null object
#GarageCond       1379 non-null object
#PavedDrive       1460 non-null object
#WoodDeckSF       1460 non-null int64
#OpenPorchSF      1460 non-null int64
#EnclosedPorch    1460 non-null int64
#3SsnPorch        1460 non-null int64
#ScreenPorch      1460 non-null int64
#PoolArea         1460 non-null int64
#PoolQC           7 non-null object
#Fence            281 non-null object
#MiscFeature      54 non-null object
#MiscVal          1460 non-null int64
#MoSold           1460 non-null int64
#YrSold           1460 non-null int64
#SaleType         1460 non-null object
#SaleCondition    1460 non-null object
#SalePrice        1460 non-null int64

#fair un mask pour les 1er Flr surface + 2eme Flr Surface == GrdLivArea

# =============================================================================

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import cm
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import manifold, datasets
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import matplotlib.font_manager
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

#############Functions ######################

def scatter_numerical_variables_and_saleprice(data, col):
    Grid_SalePrice = pd.concat([data["SalePrice"], data[col]], axis=1)
    Grid_SalePrice.plot.scatter(x=col, y='SalePrice')

def boxplot_categorical_features(data, col, figsize):
    categorical = pd.concat([data["SalePrice"], data[col]], axis=1)
    f, ax = plt.subplots(figsize=figsize)
    fig = sns.boxplot(x = col, y="SalePrice", data = categorical)
    fig.axis(ymin=0, ymax=800000)

def boxplot_categorical_features2(data,col,figsize):
    data = pd.concat([data['SalePrice'],data[col]], axis = 1)
    fig, ax = plt.subplots(figsize=figsize)
    fig = sns.boxplot(x=col, y = 'SalePrice', data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.xticks(rotation=90)

def heatmap(variable):
    f,ax = plt.subplots(figsize=(12,9))
    sns.heatmap(variable, vmax=.8, square=True)

def heatmap_zoomed(k, data, corr):
    cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':k}, \
                     yticklabels=cols.values, xticklabels=cols.values)
    plt.show()

def scatter_corr_variables(data, cols):
    sns.set()
    sns.pairplot(data[cols], size=2.5)
    plt.show()

def make_dummies(data, categorical_vars):
    for var in categorical_vars:
        data = pd.concat([data, pd.get_dummies(data[var], prefix=var)], 1)
        data = data.drop(var,1)
    return data

def fillna_and_converting_int_to_float(data):
    for i in data.columns:
        if data[i].dtype == 'int64':
            data[i] = data[i].astype('float')
    for i in data.columns:
        if data[i].dtype == 'object':
            continue
        else:
            data[i] = data[i].fillna(data[i].median())
    return data

def cols_type(data, type):
    r = []
    for i in data:
        if data[i].dtype == type:
            r.append(i)
    return r

def logistic_reg(c, x, y):
    LR = LogisticRegression(C=c)
    LR.fit(x, y)
    train_score = LR.score(x,y)
    return train_score

def get_dummies(data, cols):
    for i in cols:
        data= pd.get_dummies(data, i)
        data.drop(i, axis=1)
    return data

def tsne_function(data, n_components):
    tsne = manifold.TSNE(n_components= n_components, perplexity=30, init='random',random_state=0)
    Y = tsne.fit_transform(data)
    plt.figure()
    plt.scatter(Y[:,0], Y[:,1])
    plt.show()
    r = tsne.predict()


def functions_make_blobs(n_samples, centers, features):
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=features, random_state=0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2], c=y)
    plt.show()

def scaler_func(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def RandomForestClassifierFunction(X_train,X_test,y_train, y_test,parameters):
    acc_scorer = make_scorer(accuracy_score)
    clf = RandomForestClassifier()
    grid_obj = GridSearchCV(clf, parameters,scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, y_train)
    clf = grid_obj.best_estimator_
    clf.fit(X_train,y_train)
    prediction = clf.predict(X_test)
    print (accuracy_score(y_test,prediction))
    return accuracy_score(y_test,prediction)

#############Understanding data###############

#numeric_columns = ['GrLivArea','GarageArea','MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageArea']
#correlation = correlation["SalePrice"].abs().sort_values(ascending= False)
#scatter_numerical_variables_and_saleprice(numeric_columns)
#boxplot_categorical_features(train, 'OverallQual', (8,6))
#boxplot_categorical_features2(train,'YearBuilt', (16,8))
#heatmap_zoomed(10,train,correlation)
#cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#scatter_corr_variables(train, cols)
#missing_data(train)


train = fillna_and_converting_int_to_float(train)
# varible with all object and float type columns
string_cols = cols_type(train, object)
float_cols = cols_type(train, float)





# make dummies
f_data = get_dummies(train, string_cols)


to_drop_columns = ['Id', 'SalePrice']
X = f_data.drop(to_drop_columns, axis=1)
y = f_data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#scaler
X_train, X_test = scaler_func(X_train, X_test)

outliers = LocalOutlierFactor()
outliers = outliers.fit_predict(X)



def LinearRegressionFunc(X_train, X_test, y_train, y_test):
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_pred_train = lm.predict(X_train)
    y_pred_test = lm.predict(X_test)
    print(int(score))



LinearRegressionFunc(X_train, X_test, y_train, y_test)


#xx, yy = np.meshgrid(np.linspace(-5,5,50), np.linspace(-5,5,50))
#Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)

#########Results##############
#r_logistic_regression = logistic_reg(1,X_train, y_train))
#parameters = {"criterion": ['entropy','gini']}
#RandomForestClassifierFunction(X_train,X_test, y_train, y_test, parameters)

##########Notes#############

#faire tourner un tsne
#faire tourner un isolation forest
#regarder les données anormales
#comporer les données anormales de isolation forest avec tles erreurs de predictions
#stratified kfold
#            ensemnling

#bagging or bootsting : random forest
