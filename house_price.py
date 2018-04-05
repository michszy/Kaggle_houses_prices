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

def clean_data(data):
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

#############Understanding data###############
correlation = train.corr()

#numeric_columns = ['GrLivArea','GarageArea','MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageArea']
#correlation = correlation["SalePrice"].abs().sort_values(ascending= False)
#scatter_numerical_variables_and_saleprice(numeric_columns)
#boxplot_categorical_features(train, 'OverallQual', (8,6))
#boxplot_categorical_features2(train,'YearBuilt', (16,8))
#heatmap_zoomed(10,train,correlation)
#cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#scatter_corr_variables(train, cols)
#missing_data(train)


train = clean_data(train)





string_cols = cols_type(train, object)



def get_dummies(data, cols):
    for i in cols:
        one_hot_i = pd.get_dummies(data[i])
        data.drop(i, axis=1)
        pd.concat([data, one_hot_i])
    return data


#scaler

# make dummies
final_data = get_dummies(train, string_cols)
