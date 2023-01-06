#Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools

#Step 2: Import dataset from csv
df=pd.read_csv('news.csv')

print(df.shape)    #get shape
print(df.head())

labels=df.label    #get labels
print(labels.head())

#Step 3: Split the dataset into training and test sets
x_train,x_test,y_train,y_test=train_test_split(df['text'],labels,test_size=0.2, random_state=7)

#initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)

#Step 4: Training model on training set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)   #fit and transform train set
tfidf_test=tfidf_vectorizer.transform(x_test)         #transform test set

#initialize a PassiveAggresiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#predict on the test set and calculate accuracy 
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
rounded_score=round(score*100,2)
print(f"Accuracy:{rounded_score}%")

#build confusion matrix
print(confusion_matrix(y_test,y_pred,labels=['FAKE','REAL']))
