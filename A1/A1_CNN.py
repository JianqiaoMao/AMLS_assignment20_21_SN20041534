# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:19:04 2020

@author: Mao Jianqiao
"""
import os
import pandas as pd
import numpy as np
import time
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

#### This Code is for running on the server with multiple GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# gpus= tf.config.list_physical_devices('GPU') 
# print(gpus) 
# tf.config.experimental.set_memory_growth(gpus[0], True) 
# tf.config.experimental.set_memory_growth(gpus[1], True) 
# tf.config.experimental.set_memory_growth(gpus[2], True) 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

## Monitor the image reading process ##
def ProcessBar(count,sampleNum):
    bar='\r %.2f%% %s%s (%d processed, duration: %.2fs)'
    if (count+1)%1==0:
        duration=time.time()-start
        F='#'*int((count+1)/(sampleNum*0.025))
        nF='-'*int((sampleNum-(count+1))/(sampleNum*0.025))
        percentage=((count+1)/(sampleNum))*100
        print(bar %(percentage, F, nF, count+1,duration),end='')

#%%---------------------Data Acquisition and Preprocessing---------------------        

data_path="../Datasets/celeba"
label=pd.read_csv(data_path+"/labels.csv",sep='	', index_col=0)
img_names=list(label['img_name'])
gender=list(label['gender'])
smiling=list(label['smiling'])
gender_binary=np.array(list(map(lambda x:int((x+1)/2),gender)))

count=0
start=time.time()
dataset_A1=[]

for img_index in range(len(img_names)):
#    ProcessBar(count,len(img_names))
    count+=1
    img=cv2.imread(data_path+"/img/"+img_names[img_index])
    img_resize=cv2.resize(img,(128,128))
    img_rescale=img_resize/255.
    dataset_A1.append(img_resize)

x=np.array(dataset_A1)
X_train_Val,X_test,y_train_Val,y_test=train_test_split(x,gender_binary,test_size=0.1,random_state=10) 
#%% --------------------------Training and Validation Phase-------------------------------

acc=[]
prec=[]
f1score=[]
recall=[]

kf=KFold(5,True)
kf_num=0

for train_index, val_index in kf.split(X_train_Val):
    kf_num+=1
    x_train, x_val = X_train_Val[train_index], X_train_Val[val_index]
    y_train, y_val = y_train_Val[train_index], y_train_Val[val_index]
    
    cnn = Sequential()
    cnn.add(Conv2D(32, (8,8), input_shape = (128,128,3), activation = 'relu'))
    cnn.add(BatchNormalization())
    
    cnn.add(Conv2D(32, (8,8), activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    
    cnn.add(Conv2D(64, (5,5), activation = 'relu'))
    cnn.add(BatchNormalization())
    
    cnn.add(Conv2D(64, (5,5), activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(3,3)))
    
    cnn.add(Conv2D(128, (3,3), activation = 'relu'))
    cnn.add(BatchNormalization())
    
    cnn.add(Conv2D(128, (3,3), activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(3,3)))
    
    cnn.add(Flatten())
    
    cnn.add(Dense(units = 64, activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.3))
    
    cnn.add(Dense(units = 64, activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.3))
    
    cnn.add(Dense(units = 32, activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.2))
    
    cnn.add(Dense(units = 1, activation = 'sigmoid'))
    
    if kf_num==1:
        print(cnn.summary())
    
    cnn.compile(optimizer = 'sgd' ,loss = 'binary_crossentropy', metrics = ['accuracy']) 

    def scheduler(epoch):
        if epoch % 10 == 0 and epoch != 0:
            lr = K.get_value(cnn.optimizer.lr)
            K.set_value(cnn.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
        return K.get_value(cnn.optimizer.lr)

    early_stopping=EarlyStopping(monitor='val_accuracy', min_delta=0,
                              patience=100, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=True)
    
    reduce_lr = LearningRateScheduler(scheduler)
    record=cnn.fit(x=x_train, y=y_train,
                       epochs=60,batch_size=16,
                       callbacks=[reduce_lr,early_stopping],
                       verbose=2,validation_data=(x_val,y_val))
    
    accRec=record.history['accuracy']
    val_accRec=record.history['val_accuracy']
    lossRec=record.history['loss']
    val_lossRec=record.history['val_loss']
    
    predict=np.array(list(map(int,np.round(cnn.predict(x_val)))))

    acc.append(accuracy_score(y_val, predict))
    prec.append(precision_score(y_val,predict,average='macro'))
    f1score.append(f1_score(y_val,predict,average='macro'))
    recall.append(recall_score(y_val,predict,average='macro'))
    
    val_accRec=record.history['val_accuracy']
    print("The highest validation acc is {}".format(np.max(val_accRec)))
    
aveAcc=sum(acc)/len(acc)
avePrec=sum(prec)/len(prec)
avef1score=sum(f1score)/len(f1score)
aveRecall=sum(recall)/len(recall)
print("CNN 5-Fold CrossVal Acc: %.4f" %(aveAcc))
print("CNN 5-Fold CrossVal Precision: %.4f" %(avePrec))
print("CNN 5-Fold CrossVal recall: %.4f" %(aveRecall))   
print("CNN 5-Fold CrossVal f1-score: %.4f" %(avef1score))

epochs=range(1,len(accRec)+1)
plt.figure()
plt.plot(epochs,val_accRec,'b',label='validation acc',color='orange')
plt.plot(epochs,accRec,'b',label='training acc',color='blue')

plt.legend()
plt.figure()

plt.plot(epochs,lossRec,'b',label='training loss',color='blue')
plt.plot(epochs,val_lossRec,'b',label='validation loss',color='orange')
plt.legend()
plt.show()    

 #%% --------------------------Testing Phase-------------------------------
predict_test=np.array(list(map(int,np.round(cnn.predict(X_test)))))

print("CNN(S) Test Acc: %.4f" %(accuracy_score(y_test, predict_test)))
print("CNN(S) Test Precision: %.4f" %(precision_score(y_test,predict_test)))
print("CNN(S) Test recall: %.4f" %(recall_score(y_test,predict_test)))   
print("CNN(S) Test f1-score: %.4f" %(f1_score(y_test,predict_test)))

confMat=confusion_matrix(y_test, predict)
confMat=pd.DataFrame(confMat)
confMat=confMat.rename(index={0:"Female",1:"Male"},columns={0:"Female",1:"Male"})
plt.figure(num='Confusion Matrix', facecolor='lightgray')
plt.title('Confusion Matrix', fontsize=20)
ax=sns.heatmap(confMat, fmt='d',cmap='Greys',annot=True)
ax.set_xlabel('Predicted Class', fontsize=14)
ax.set_ylabel('True Class', fontsize=14)
plt.show()

    