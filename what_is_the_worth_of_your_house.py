

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
    plt.show()

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

def heatmap(data):
    corr = data.corr()
    f,ax = plt.subplots(figsize=(12,9))
    sns.heatmap(corr, vmax=.8, square=True)
    plt.show()

def heatmap2(data):
    corr = data.corr()
    f,ax = plt.subplots(figsize=(12,9))
    sns.clustermap(corr, vmax=.8, square=True)
    plt.show()

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

def logistic_reg(x, y):
    LR = LogisticRegression()
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



# varible with all object and float type columns


# make dummies

#print(len(f_data))

#X = train.drop('SalePrice', axis=1)
#y = train['SalePrice']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/len(X)))



#cols_list = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF'\
#,'1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt']
train = fillna_and_converting_int_to_float(train)
cols_list = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF'\
,'1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','SalePrice']

train = train[cols_list]


#new_x= pd.DataFrame(d)
#print(new_x)

train = fillna_and_converting_int_to_float(train)
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/len(X)))


def logistic_reg(x):
    LR = LogisticRegression()
    LR.fit(X_train, y_train)
    r = LR.predict(x)
    return r



def welcome_dear_user():

    print('Welcome dear user please enter your name')
    name = input()
    print("Hello {}, welcome to the house prediction program".format(name))
    print("How about we predict the price of your house??")
    while True:
        print("Y/N")
        ok = input()
        if (ok == "Y"):
            question_asked_process(name)
            print('Thank you for using our program')
            v = input()
        elif (ok=='N'):
            print('Ok nevermind ... ')
            break
        else:
            print("Sorry I didn't understand")

#cols_list = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF'\
#,'1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','SalePrice']



def question_asked_process(name):
    x = []
    question_list = [["Rates the overall material and finish of the house (1 to 10)"], \
                ['Above grade (ground) living area square feet'], \
                ['Size of garage in car capacity'], \
                ['Size of garage in square feet'], \
                ['Total square feet of basement area'], \
                ['First Floor square feet'], \
                ['Number of Bathroom'], \
                ['Number of rooms'], \
                ['Original construction date']]
    for i in question_list:
        print (i)
        v = float(input())
        x.append(v)
    x = pd.DataFrame([question_list, x])
    x = x.drop(0)
    print("Your house is worth $ {}".format(logistic_reg(x)))
    return x




welcome_dear_user()





##########Notes#############
#user_welcome()

#faire tourner un tsne
#faire tourner un isolation forest
#regarder les données anormales
#comporer les données anormales de isolation forest avec tles erreurs de predictions
#stratified kfold
#            ensemnling

#bagging or bootsting : random forest
