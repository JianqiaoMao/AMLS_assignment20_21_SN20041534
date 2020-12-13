# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:10:24 2020

@author: Mao Jianqiao
"""
import os
import pandas as pd
import numpy as np
import time
import face_recognition
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

## Monitor the image reading process ##
def ProcessBar(count,sampleNum,startTime,outliers):
    bar='\r %.2f%% %s%s (%d processed, duration: %.2fs, %d outliers detected)'
    if (count+1)%1==0:
        duration=time.time()-startTime
        F='#'*int((count+1)/(sampleNum*0.025))
        nF='-'*int((sampleNum-(count+1))/(sampleNum*0.025))
        percentage=((count+1)/(sampleNum))*100
        outlinerNum=len(outliers)
        print(bar %(percentage, F, nF, count+1,duration,outlinerNum),end='')
  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
data_path="../Datasets/celeba"
label=pd.read_csv(data_path+"/labels.csv",sep='	', index_col=0)
img_names=list(label['img_name'])
gender=list(label['gender'])
gender_binary=np.array(list(map(lambda x:int((x+1)/2),gender)))

count=0
start=time.time()
dataset=np.zeros((len(img_names),128))
outlier_index=[]

for img_index in range(len(img_names)):
    ProcessBar(count,len(img_names),start,outlier_index)
    count+=1
    img = face_recognition.load_image_file(data_path+"/img/"+img_names[img_index])
    face_encoding = face_recognition.face_encodings(img)
    if len(face_encoding)==0:
        outlier_index.append(img_index)
    else:
        dataset[img_index]=face_encoding[0]

dataset=np.delete(dataset, outlier_index, 0)
gender_binary=np.delete(gender_binary, outlier_index, 0)

X_train,X_test,y_train,y_test=train_test_split(dataset,gender_binary,test_size=0.1,random_state=3) 

std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)

classifiers=[KNeighborsClassifier(),svm.SVC(),AdaBoostClassifier(),BaggingClassifier()]
classifierNames=['KNN','SVM','Boosting','Bagging']
parameters=[{'n_neighbors':[3,5,7,9,11]},
            {'kernel':['rbf'],'C':[0.5,0.9,2]},
            {'n_estimators':[50,150,300,500]},
            {'n_estimators':[200,500,800,1500], 'max_features':[12,16,20],'max_samples':[0.3,0.5]}]

for i in range(len(classifiers)):
   clf=classifiers[i]
   print('\n parameters of '+classifierNames[i]+' is searching...')
   clf_candidate= GridSearchCV(clf, parameters[i], scoring = 'accuracy',cv = 5)
   result = clf_candidate.fit(X_train_std,y_train)
   bestModel=clf_candidate.best_estimator_
  
   print("The best para. for "+classifierNames[i]+" is {}".format(clf_candidate.best_params_))
   print("Best validation acc: %f" % (result.best_score_))

   pred=bestModel.predict(X_test_std)
  
   acc=accuracy_score(y_test, pred)
   prec=precision_score(y_test,pred)
   recall=recall_score(y_test,pred)
   f1=f1_score(y_test,pred)
  
   print(str(classifierNames[i])+" Test Acc: %.4f" %(acc))
   print(str(classifierNames[i])+" Test Precision: %.4f" %(prec))
   print(str(classifierNames[i])+" Test recall: %.4f" %(recall))
   print(str(classifierNames[i])+" Test f1-score: %.4f" %(f1))