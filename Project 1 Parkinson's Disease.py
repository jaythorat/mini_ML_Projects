#Step 1: Importing necessary libraries
import numpy as np 
import pandas as pd 
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Step 2: Import datasets from csv
df=pd.read_csv('parkinson.csv')
df.head()
print(df.head())

features=df.loc[:,df.columns!='status'].values[:,1:]   #get features
labels=df.loc[:,'status'].values                              #get labels

# get the count of each label (0 and 1) in labels
print(labels[labels==1].shape[0], labels[labels==0].shape[0])

#scale the features to between -1 and 1
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

#Step 3: Splitting the dataset into training and test sets
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=7)

#Step 4: Training the model on training set
model=XGBClassifier()
model.fit(x_train,y_train)

#calculate the accuracy
y_pred=model.predict(x_test)  #predicted values for x
print(accuracy_score(y_test,y_pred)*100)