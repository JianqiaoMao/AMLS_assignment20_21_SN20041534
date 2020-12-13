# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:22:10 2020

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
smiling=list(label['smiling'])
smiling_binary=np.array(list(map(lambda x:int((x+1)/2),smiling)))

count=0
start=time.time()
key_features=['bottom_lip','top_lip','left_eye','right_eye','left_eyebrow','right_eyebrow']

dataset=np.zeros((len(img_names),92))
outlier_index=[]

for img_index in range(len(img_names)):
    ProcessBar(count,len(img_names),start,outlier_index)
    count+=1
    img = face_recognition.load_image_file(data_path+"/img/"+img_names[img_index])
    face_landmarks = face_recognition.face_landmarks(img)
    if len(face_landmarks)==0:
        outlier_index.append(img_index)
    else:
        point_index=0
        for feature in key_features:
            for point in face_landmarks[0][feature]:
                dataset[img_index,point_index]=point[0]
                dataset[img_index,point_index+1]=point[1]
                point_index+=2

dataset=np.delete(dataset, outlier_index, 0)
smiling_binary=np.delete(smiling_binary, outlier_index, 0)

X_train,X_test,y_train,y_test=train_test_split(dataset,smiling_binary,test_size=0.1,random_state=3) 

std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)

classifiers=[KNeighborsClassifier(),svm.SVC(),AdaBoostClassifier(),BaggingClassifier()]
classifierNames=['KNN','SVM','Boosting','Bagging']
parameters=[{'n_neighbors':[5,10,15,20]},
            {'kernel':['rbf','poly'],'C':[0.1,1,2,5]},
            {'n_estimators':[50,100,200,500]},
            {'n_estimators':[200,500,1000], 'max_features':[12,16,20],'max_samples':[0.3,0.5]}]

for i in range(len(classifiers)):
   clf=classifiers[i]
   print('\n parameters of '+classifierNames[i]+' is searching...')
   clf_candidate= GridSearchCV(clf,parameters[i], scoring = 'accuracy',cv = 5)
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
