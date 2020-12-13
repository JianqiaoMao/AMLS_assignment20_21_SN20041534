# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:45:22 2020

@author: Mao Jianqiao
"""
import os
import pandas as pd
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

## Monitor the image reading process ##
def ProcessBar(count,sampleNum,startTime):
    bar='\r %.2f%% %s%s (%d processed, duration: %.2fs.)'
    if (count+1)%1==0:
        duration=time.time()-startTime
        F='#'*int((count+1)/(sampleNum*0.025))
        nF='-'*int((sampleNum-(count+1))/(sampleNum*0.025))
        percentage=((count+1)/(sampleNum))*100
        print(bar %(percentage, F, nF, count+1,duration),end='')   
#%% ---------------------Data Acquisition and Preprocessing---------------------    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
data_path="../Datasets/cartoon_set"
label=pd.read_csv(data_path+"/labels.csv",sep='	', index_col=0)
img_names=list(label['file_name'])
shape=np.array(list(label['face_shape']))

count=0
start=time.time()

dataset=np.zeros((len(img_names),28*38))

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
    dataset[img_index,:] = flatten

X_train_Val,X_test,y_train_Val,y_test=train_test_split(dataset,shape,test_size=0.1,random_state=3) 

#%% --------------------------Training and Validation Phase-------------------------------
kf=KFold(5,True)

valAcc=[]
valPrec=[]
valf1score=[]
valRecall=[]
print("\n5 Fold Cross Validation is used.")
for train_index, test_index in kf.split(X_train_Val):
    X_train, X_val = X_train_Val[train_index], X_train_Val[test_index]
    y_train, y_val = y_train_Val[train_index], y_train_Val[test_index]
    
    KNN=KNeighborsClassifier(n_neighbors=8) 
    KNN.fit(X_train,y_train)
    
    pred_val=KNN.predict(X_val)

    valAcc.append(accuracy_score(y_val, pred_val))
    valPrec.append(precision_score(y_val, pred_val,average='macro'))
    valRecall.append(recall_score(y_val, pred_val,average='macro'))
    valf1score.append(f1_score(y_val, pred_val,average='macro'))

aveValAcc=sum(valAcc)/len(valAcc)
aveValPrec=sum(valPrec)/len(valPrec)
aveValRecall=sum(valRecall)/len(valRecall)
aveValf1score=sum(valf1score)/len(valf1score)

print("KNN 5-Fold CV average Acc: %.4f" %(aveValAcc))
print("KNN 5-Fold CV average Precision: %.4f" %(aveValPrec))
print("KNN 5-Fold CV average recall: %.4f" %(aveValRecall))
print("KNN 5-Fold CV average f1-score: %.4f \n" %(aveValf1score))

#%% --------------------------Testing Phase-------------------------------
pred_test = KNN.predict(X_test)
testAcc = accuracy_score(y_test, pred_test)
testPrec = precision_score(y_test, pred_test,average='macro')
testRecall = recall_score(y_test, pred_test,average='macro')
testf1score = f1_score(y_test, pred_test,average='macro')

print("{} instances are used for testing.".format(len(y_test)))
print("KNN Test Acc: %.4f" %(testAcc))
print("KNN Test Precision: %.4f" %(testPrec))
print("KNN Test recall: %.4f" %(testRecall))
print("KNN Test f1-score: %.4f" %(testf1score))

confMat=confusion_matrix(y_test, pred_test)
confMat=pd.DataFrame(confMat)
plt.figure(num='Confusion Matrix B1', facecolor='lightgray')
plt.title('Confusion Matrix B1', fontsize=20)
ax=sns.heatmap(confMat, fmt='d',cmap='Greys',annot=True)
ax.set_xlabel('Predicted Class', fontsize=14)
ax.set_ylabel('True Class', fontsize=14)
plt.show()