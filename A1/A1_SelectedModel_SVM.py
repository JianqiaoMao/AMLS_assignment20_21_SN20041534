# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:45:22 2020

@author: Mao Jianqiao
"""
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import face_recognition
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix

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
#%% ---------------------Data Acquisition and Preprocessing---------------------    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
data_path="../Datasets/celeba"
label=pd.read_csv(data_path+"/labels.csv",sep='	', index_col=0)
img_names=list(label['img_name'])
gender=list(label['gender'])
gender_binary=np.array(list(map(lambda x:int((x+1)/2),gender)))

count=0
start=time.time()
key_features=['bottom_lip','top_lip','left_eye','right_eye','chin','nose_bridge','nose_tip','left_eyebrow','right_eyebrow']

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

X_train_Val,X_test,y_train_Val,y_test=train_test_split(dataset,gender_binary,
                                                       test_size=0.1,random_state=3) 

#%% --------------------------Training and Validation Phase-------------------------------
kf=KFold(5,True)

valAcc=[]
valPrec=[]
valf1score=[]
valRecall=[]
print("5 Fold Cross Validation is used.")
for train_index, test_index in kf.split(X_train_Val):
    X_train, X_val = X_train_Val[train_index], X_train_Val[test_index]
    y_train, y_val = y_train_Val[train_index], y_train_Val[test_index]

    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    X_val_std = std.transform(X_val)
  
    SVM=svm.SVC(C=0.9,kernel='rbf')    
    SVM.fit(X_train,y_train)
    
    pred_val=SVM.predict(X_val)

    valAcc.append(accuracy_score(y_val, pred_val))
    valPrec.append(precision_score(y_val, pred_val))
    valRecall.append(recall_score(y_val, pred_val))
    valf1score.append(f1_score(y_val, pred_val))

aveValAcc=sum(valAcc)/len(valAcc)
aveValPrec=sum(valPrec)/len(valPrec)
aveValRecall=sum(valRecall)/len(valRecall)
aveValf1score=sum(valf1score)/len(valf1score)

print("SVM 5-Fold CV average Acc: %.4f" %(aveValAcc))
print("SVM 5-Fold CV average Precision: %.4f" %(aveValPrec))
print("SVM 5-Fold CV average recall: %.4f" %(aveValRecall))
print("SVM 5-Fold CV average f1-score: %.4f \n" %(aveValf1score))

#%% --------------------------Testing Phase-------------------------------
X_test_std = std.transform(X_test)

pred_test = SVM.predict(X_test)
testAcc = accuracy_score(y_test, pred_test)
testPrec = precision_score(y_test, pred_test)
testRecall = recall_score(y_test, pred_test)
testf1score = f1_score(y_test, pred_test)

print("{} instances are used for testing.".format(len(y_test)))
print("SVM Test Acc: %.4f" %(testAcc))
print("SVM Test Precision: %.4f" %(testPrec))
print("SVM Test recall: %.4f" %(testRecall))
print("SVM Test f1-score: %.4f" %(testf1score))

confMat=confusion_matrix(y_test, pred_test)
confMat=pd.DataFrame(confMat)
confMat=confMat.rename(index={0:'female',1:'male'},columns={0:'female',1:'male'})
plt.figure(num='Confusion Matrix A1', facecolor='lightgray')
plt.title('Confusion Matrix A1', fontsize=20)
ax=sns.heatmap(confMat, fmt='d',cmap='Greys',annot=True)
ax.set_xlabel('Predicted Class', fontsize=14)
ax.set_ylabel('True Class', fontsize=14)
plt.show()