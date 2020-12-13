# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:13:48 2020

@author: Mao Jianqiao
"""
import os
import pandas as pd
import numpy as np
import time
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

## Monitor the image reading process ##
def ProcessBar(count,sampleNum,startTime):
    bar='\r %.2f%% %s%s (%d processed, duration: %.2fs)'
    if (count+1)%1==0:
        duration=time.time()-startTime
        F='#'*int((count+1)/(sampleNum*0.025))
        nF='-'*int((sampleNum-(count+1))/(sampleNum*0.025))
        percentage=((count+1)/(sampleNum))*100
        print(bar %(percentage, F, nF, count+1,duration),end='')

print("program start!")

data_path = '../Datasets/cartoon_set'
label=pd.read_csv(data_path+"/labels.csv",sep='	', index_col=0)
img_names=list(label['file_name'])
shape=label['face_shape']

count=0
start=time.time()
key_features=['chin']

rescale_size=32
dataset_B1=np.zeros((len(img_names),28*38))

for img_index in range(len(img_names)):
    ProcessBar(count,len(img_names),start)
    count+=1
    img=cv2.imread(data_path+"/img/"+img_names[img_index])
    shapeSeg=img[270:410,155:345,:]
    gray = cv2.cvtColor(shapeSeg,cv2.COLOR_BGR2GRAY) 
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    xgrad = cv2.Sobel(blurred, cv2.CV_16SC1, 1, 0)
    ygrad = cv2.Sobel(blurred, cv2.CV_16SC1, 0, 1)
    edge= cv2.Canny(xgrad, ygrad, 50, 150)
    resize = cv2.resize(edge,(38,28))
    norm = resize/255.
    flatten = norm.flatten()
    dataset_B1[img_index,:] = flatten

X_train,X_test,y_train,y_test=train_test_split(dataset_B1,shape,test_size=0.1,random_state=3) 

classifiers=[KNeighborsClassifier(),svm.SVC(),AdaBoostClassifier(),BaggingClassifier()]
classifierNames=['KNN','SVM','Boosting','Bagging']
parameters=[{'n_neighbors':[5,8,15,20]},
            {'kernel':['rbf','poly'],'C':[0.1,1,2,5]},
            {'n_estimators':[100,200,500,1000]},
            {'n_estimators':[200,500,1000,2000], 'max_features':[5,10,15,20],'max_samples':[0.3,0.5]}]

for i in range(len(classifiers)):
   clf=classifiers[i]
   print('\n parameters of '+classifierNames[i]+' is searching...')
   clf_candidate= GridSearchCV(clf,parameters[i], scoring = 'accuracy',cv = 5)
   result = clf_candidate.fit(X_train,y_train)
   bestModel=clf_candidate.best_estimator_
  
   print("The best para. for "+classifierNames[i]+" is {}".format(clf_candidate.best_params_))
   print("Best validation acc: %f" % (result.best_score_))
  
   pred=bestModel.predict(X_test)
  
   acc=accuracy_score(y_test, pred)
   prec=precision_score(y_test,pred,average='macro')
   recall=recall_score(y_test,pred,average='macro')
   f1=f1_score(y_test,pred,average='macro')
  
   print(str(classifierNames[i])+" Acc: %.4f" %(acc))
   print(str(classifierNames[i])+" Precision: %.4f" %(prec))
   print(str(classifierNames[i])+" recall: %.4f" %(recall))
   print(str(classifierNames[i])+" f1-score: %.4f" %(f1))