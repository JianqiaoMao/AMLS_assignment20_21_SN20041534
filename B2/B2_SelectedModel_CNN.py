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
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
## train on GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
color=np.array(list(label['eye_color']))

count=0
start=time.time()

dataset=np.zeros((len(img_names),50,200,4))

for img_index in range(len(img_names)):
    ProcessBar(count,len(img_names),start)
    count+=1
    img = mpimg.imread(data_path+"/img/"+img_names[img_index])
    img_colorSeg=img[240:290,150:350,:]
    norm_color = img_colorSeg/255.
    dataset[img_index,:] = norm_color

X_train_Val,X_test,y_train_Val,y_test=train_test_split(dataset,color,test_size=0.1,random_state=3) 

#%% --------------------------Training and Validation Phase-------------------------------
kf=KFold(8,True)
kf_num=0

valAcc=[]
valPrec=[]
valf1score=[]
valRecall=[]
print("8 Fold Cross Validation is used.")
for train_index, test_index in kf.split(X_train_Val):  
    kf_num+=1
    
    X_train, X_val = X_train_Val[train_index], X_train_Val[test_index]
    y_train, y_val = y_train_Val[train_index], y_train_Val[test_index]
    y_train = to_categorical(y_train, num_classes=5)
    y_val = to_categorical(y_val, num_classes=5)
    
    cnn = Sequential()
    cnn.add(Conv2D(32, (2,8), input_shape = (50,200,4), activation = 'relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2,4)))

    cnn.add(Conv2D(64, (3,6), activation = 'relu'))
    cnn.add(BatchNormalization()) 
    cnn.add(MaxPooling2D(pool_size=(2,4)))

    cnn.add(Conv2D(64, (3,3), activation = 'relu'))
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

    cnn.compile(optimizer = 'adam' ,
                loss = 'categorical_crossentropy', 
                metrics = ['accuracy'])
    
    def scheduler(epoch):
        if epoch % 15 == 0 and epoch != 0:
            lr = K.get_value(cnn.optimizer.lr)
            K.set_value(cnn.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(cnn.optimizer.lr)
    
    reduce_lr = LearningRateScheduler(scheduler)
    early_stopping=EarlyStopping(monitor='val_accuracy', min_delta=0,
                              patience=7, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=True)
    record=cnn.fit(x=X_train, y=y_train,
                   epochs=70,batch_size=32,
                   callbacks=[reduce_lr,early_stopping],
                   verbose=2,
                   validation_data=(X_val,y_val))    

    accRec=record.history['accuracy']
    val_accRec=record.history['val_accuracy']
    lossRec=record.history['loss']
    val_lossRec=record.history['val_loss']
    print("The highest validation acc in round{} is {}".format(kf_num,np.max(val_accRec))) 

    pred_val=np.argmax(cnn.predict(X_val),axis=1)
    y_val=np.argmax(y_val,axis=1)
    
    valAcc.append(accuracy_score(y_val, pred_val))
    valPrec.append(precision_score(y_val, pred_val,average='macro'))
    valRecall.append(recall_score(y_val, pred_val,average='macro'))
    valf1score.append(f1_score(y_val, pred_val,average='macro'))   

aveValAcc=sum(valAcc)/len(valAcc)
aveValPrec=sum(valPrec)/len(valPrec)
aveValRecall=sum(valRecall)/len(valRecall)
aveValf1score=sum(valf1score)/len(valf1score)

print("CNN 8-Fold CV average Acc: %.4f" %(aveValAcc))
print("CNN 8-Fold CV average Precision: %.4f" %(aveValPrec))
print("CNN 8-Fold CV average recall: %.4f" %(aveValRecall))
print("CNN 8-Fold CV average f1-score: %.4f \n" %(aveValf1score))

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
pred_test = np.argmax(cnn.predict(X_test),axis=1)

testAcc = accuracy_score(y_test, pred_test)
testPrec = precision_score(y_test, pred_test,average='macro')
testRecall = recall_score(y_test, pred_test,average='macro')
testf1score = f1_score(y_test, pred_test,average='macro')

print("{} instances are used for testing.".format(len(y_test)))
print("CNN Test Acc: %.4f" %(testAcc))
print("CNN Test Precision: %.4f" %(testPrec))
print("CNN Test recall: %.4f" %(testRecall))
print("CNN Test f1-score: %.4f" %(testf1score))

confMat=confusion_matrix(y_test, pred_test)
confMat=pd.DataFrame(confMat)
plt.figure(num='Confusion Matrix', facecolor='lightgray')
plt.title('Confusion Matrix CNN', fontsize=20)
ax=sns.heatmap(confMat, fmt='d',cmap='Greys',annot=True)
ax.set_xlabel('Predicted Class', fontsize=14)
ax.set_ylabel('True Class', fontsize=14)
plt.show()

cnn.save('CNN.h5')