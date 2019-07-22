# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:20:55 2019

@author: erasunn
"""

'''
  @data !wget -O teleCust1000t.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import metrics

def TrainTestSplit(xData, yData):
    X_train, X_test, y_train, y_test = train_test_split(xData, yData, test_size=0.2, random_state=4)
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)
    return X_train, X_test, y_train, y_test

def TrainingModel(k, x_train, y_tarin):
    Knc = KNeighborsClassifier(n_neighbors=k)
    #print(type(x_train))
    #print(type(ydata))
    Knc.fit(xdata, ydata)  
    return Knc
    
def TestAndEvaluation(Knc, x_test, y_test, x_train):
    y_hat = Knc.predict(x_test)
    y_pre_proba = Knc.predict_proba(x_test)
    #print ("probability vector ", y_pre_proba)
    mean_score = Knc.score(x_test, y_test)
    print("Train set Accuracy: ", metrics.accuracy_score(y_train, Knc.predict(x_train)))
    print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_hat))
    print ("Mean Square error", mean_score)
    

######## MAIN ###############
df = pd.read_csv(r'C:\Users\ERASUNN\Downloads\teleCust1000t.csv')
#print (df.head(9))
print (df.describe())
xdata = df[['region', 'tenure','age', 'income', 'ed', 'employ']].values
ydata = df['custcat'].values
#print (df['tenure'].value_counts())
X = preprocessing.StandardScaler().fit(xdata).transform(xdata.astype(float))
#print(X[0:5])
x_train, x_test, y_train, y_test = TrainTestSplit(X, ydata)
plt.scatter(x_train['income'], x_train['age'], color = 'red')
plt.show()
for i in range (1,10):
    Knc = TrainingModel(i, x_train, y_train)
    TestAndEvaluation(Knc, x_test, y_test, x_train)
#print ("DataSet ", X_train, y_test)

