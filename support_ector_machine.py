import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix,classification_report


dataset=pd.read_csv(r"C:\Users\hp\Downloads\logit classification.csv")

x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

#split the data test size 20%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)#incresed random state but accuracy never goes to incresed so 
#keep random state=0


'''
#we can see withaout feature scalar we cant get goog accuracy
from sklearn.preprocessing import Normalizer
sc=Normalizer()
x_train=sc.fit_transform(x_train) #fit is bydefault in train
x_test=sc.transform(x_test)  
#we remove fit from test to increase accuracy'''

#training the svm model
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

#create prediction for x_test
y_pred=classifier.predict(x_test)


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
print(cm)

#model accuracy
from sklearn.metrics import accuracy_score #tp+tn/tp+tn+fp+fn,accuracy
ac=accuracy_score(y_test,y_pred)
print(ac)


#trainig accuracy
bias=classifier.score(x_train,y_train)  #bias=training 
print(bias)

#testing accuracy

variance=classifier.score(x_test,y_test) #variance=testing 
print(variance)

'''
with standard scaler & using Bernouli algorithm--accuracy-0.825
with standard scaler & using GAusian algorithm--acciracy-0.9125
with standard scaler & using Multinomial algorithm--error
with Normalizer & using Multinomial algorithm--accuracy--0.725
without scaling & using Multinomial algorithm--accuracy--0.5625
without scaling & using Bernouli algorithm--accuracy-0.725

'''