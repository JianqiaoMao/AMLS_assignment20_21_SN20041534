# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:13:48 2020

@author: Mao Jianqiao
"""
import pandas as pd
import numpy as np
import time

import cv2

import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
from keras.utils import to_categorical
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns

#### This Code is for running on the server with multiple GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# gpus= tf.config.list_physical_devices('GPU') 
# print(gpus) 
# tf.config.experimental.set_memory_growth(gpus[0], True) 
# tf.config.experimental.set_memory_growth(gpus[1], True) 
# tf.config.experimental.set_memory_growth(gpus[2], True) 
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# print(tf.test.gpu_device_name())
#GPUtil.showUtilization(all=True)

## Monitor the image reading process ##
def ProcessBar(count,sampleNum,startTime):
    bar='\r %.2f%% %s%s (%d processed, duration: %.2fs)'
    if (count+1)%1==0:
        duration=time.time()-startTime
        F='#'*int((count+1)/(sampleNum*0.025))
        nF='-'*int((sampleNum-(count+1))/(sampleNum*0.025))
        percentage=((count+1)/(sampleNum))*100
        print(bar %(percentage, F, nF, count+1,duration),end='')
#%%---------------------Data Acquisition and Preprocessing---------------------

data_path="D:/Happiness source/MS/Applied ML I/course Proj/AMLS_Project/Datasets/cartoon_set"
label=pd.read_csv(data_path+"/labels.csv",sep='	', index_col=0)
img_names=list(label['file_name'])
shape=np.array(label['face_shape'])

count=0
start=time.time()
dataset_B1=np.zeros((len(img_names),67,95,4))

for img_index in range(len(img_names)):
    ProcessBar(count,len(img_names),start)
    count+=1
    img = mpimg.imread(data_path+"/img/"+img_names[img_index])
    img_shapeSeg=img[260:395,155:345,:]
    norm_color = img_shapeSeg/255.
    resize = cv2.resize(img,(95,67))
    dataset_B1[img_index,:] = resize

x=dataset_B1
X_train_Val,X_test,y_train_Val,y_test=train_test_split(x,shape,test_size=0.1,random_state=10) 
#%% --------------------------Training and Validation Phase-------------------------------
acc_val=[]
prec_val=[]
f1score_val=[]
recall_val=[]

kf=KFold(5,True)
kf_num=0

for train_index, val_index in kf.split(X_train_Val):
    kf_num+=1
    x_train, x_val = X_train_Val[train_index], X_train_Val[val_index]
    y_train, y_val = y_train_Val[train_index], y_train_Val[val_index]
    y_train = to_categorical(y_train, num_classes=5)
    y_val = to_categorical(y_val, num_classes=5)
    
    cnn = Sequential()
    cnn.add(Conv2D(32, (6,8), input_shape = (67,95,4), activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(4,5)))
    
    cnn.add(Conv2D(64, (4,6), activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(3,3)))  
      
    cnn.add(Flatten()) 

    cnn.add(Dense(units = 64, activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.4))

    cnn.add(Dense(units = 32, activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.3))

    cnn.add(Dense(units = 5, activation = 'softmax'))
    
    if kf_num==1:
        print(cnn.summary())
 
    cnn.compile(optimizer = 'adam' ,loss = 'categorical_crossentropy', metrics = ['accuracy'])

    def scheduler(epoch):
        if epoch % 10 == 0 and epoch != 0:
            lr = K.get_value(cnn.optimizer.lr)
            K.set_value(cnn.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(cnn.optimizer.lr)

    early_stopping=EarlyStopping(monitor='val_loss', min_delta=0,
                              patience=5, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=True)
    
    reduce_lr = LearningRateScheduler(scheduler)
    record=cnn .fit(x=x_train, y=y_train,epochs=30,batch_size=16,callbacks=[reduce_lr,early_stopping],verbose=2,validation_data=(x_val,y_val))
        
    accRec=record.history['accuracy']
    val_accRec=record.history['val_accuracy']
    lossRec=record.history['loss']
    val_lossRec=record.history['val_loss']

    print("The highest validation acc is {}".format(np.max(val_accRec)))
    
    predict=np.argmax(cnn.predict(x_val),axis=1)
    y_val=np.argmax(y_val,axis=1)

    acc_val.append(accuracy_score(y_val, predict))
    prec_val.append(precision_score(y_val,predict,average='macro'))
    f1score_val.append(f1_score(y_val,predict,average='macro'))
    recall_val.append(recall_score(y_val,predict,average='macro'))
    
aveAcc=sum(acc_val)/len(acc_val)
avePrec=sum(prec_val)/len(prec_val)
avef1score=sum(f1score_val)/len(f1score_val)
aveRecall=sum(recall_val)/len(recall_val)
print("CNN(S) 5-Fold CrossVal Acc: %.4f" %(aveAcc))
print("CNN(S) 5-Fold CrossVal Precision: %.4f" %(avePrec))
print("CNN(S) 5-Fold CrossVal recall: %.4f" %(aveRecall))   
print("CNN(S) 5-Fold CrossVal f1-score: %.4f" %(avef1score))

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
predict_test = np.argmax(cnn.predict(X_test),axis=1)

print("CNN(S) Test Acc: %.4f" %(accuracy_score(y_test, predict_test)))
print("CNN(S) Test Precision: %.4f" %(precision_score(y_test,predict_test,average='macro')))
print("CNN(S) Test recall: %.4f" %(recall_score(y_test,predict_test,average='macro')))   
print("CNN(S) Test f1-score: %.4f" %(f1_score(y_test,predict_test,average='macro')))

confMat=confusion_matrix(y_test, predict_test)
confMat=pd.DataFrame(confMat)
plt.figure(num='Confusion Matrix', facecolor='lightgray')
plt.title('Confusion Matrix', fontsize=20)
ax=sns.heatmap(confMat, fmt='d',cmap='Greys',annot=True)
ax.set_xlabel('Predicted Class', fontsize=14)
ax.set_ylabel('True Class', fontsize=14)
plt.show()