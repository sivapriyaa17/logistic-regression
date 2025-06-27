import pandas as py
import numpy as ny
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, classification_report
d=load_breast_cancer()
x,y=d.data,d.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
m=LogisticRegression()
m.fit(x_train,y_train)
pred=m.predict(x_test)
proba=m.predict_proba(x_test)[:,1]
print("precision:",precision_score(y_test,pred))
print("recall score:",recall_score(y_test,pred))
print("ROC-AUC SCORE:",roc_auc_score(y_test,pred))
t=0.3
custom=(proba>=t).astype(int)
print(confusion_matrix(y_test,custom))

