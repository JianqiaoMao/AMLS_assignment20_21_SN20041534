# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 23:53:35 2020

@author: Mao Jianqiao
"""
import pandas as pd
import numpy as np
import time
import face_recognition
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.utils import to_categorical
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping

def ProcessBar(count,sampleNum,startTime):
    bar='\r %.2f%% %s%s (%d processed, duration: %.2fs)'
    if (count+1)%1==0:
        duration=time.time()-startTime
        F='#'*int((count+1)/(sampleNum*0.025))
        nF='-'*int((sampleNum-(count+1))/(sampleNum*0.025))
        percentage=((count+1)/(sampleNum))*100
        print(bar %(percentage, F, nF, count+1,duration),end='')  

def form_dataset(path,task):
    if task=='A1':
        label=pd.read_csv(path+"/labels.csv",sep='	', index_col=0)
        img_names=list(label['img_name'])
        gender=list(label['gender'])
        gender_binary=np.array(list(map(lambda x:int((x+1)/2),gender)))
        
        count=0
        start=time.time()
        
        dataset=np.zeros((len(img_names),128))
        outlier_index=[]
        
        print('loading and embedding faces...')
        for img_index in range(len(img_names)):
            ProcessBar(count,len(img_names),start)
            count+=1
            img = face_recognition.load_image_file(path+"/img/"+img_names[img_index])
            face_encoding = face_recognition.face_encodings(img)
            if len(face_encoding)==0:
                outlier_index.append(img_index)
            else:
                dataset[img_index]=face_encoding[0]
        
        dataset=np.delete(dataset, outlier_index, 0)
        gender_binary=np.delete(gender_binary, outlier_index, 0)
        
        return dataset, gender_binary
        
    if task == 'A2':
        
        label=pd.read_csv(path+"/labels.csv",sep='	', index_col=0)
        img_names=list(label['img_name'])
        smiling=list(label['smiling'])
        smiling_binary=np.array(list(map(lambda x:int((x+1)/2),smiling)))
        
        count=0
        start=time.time()
        key_features=['bottom_lip','top_lip','left_eye','right_eye','left_eyebrow','right_eyebrow']
        
        dataset=np.zeros((len(img_names),92))
        outlier_index=[]
        
        print('loading faces and estimating face landmarks...')
        for img_index in range(len(img_names)):
            ProcessBar(count,len(img_names),start)
            count+=1
            img = face_recognition.load_image_file(path+"/img/"+img_names[img_index])
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
        
        return dataset, smiling_binary   
    
    if task == 'B1':
        
        label=pd.read_csv(path+"/labels.csv",sep='	', index_col=0)
        img_names=list(label['file_name'])
        shape=np.array(list(label['face_shape']))
        
        count=0
        start=time.time()
        
        dataset=np.zeros((len(img_names),28*38))
        
        print('loading faces and extracting edges...')
        for img_index in range(len(img_names)):
            ProcessBar(count,len(img_names),start)
            count+=1
            img=cv2.imread(path+"/img/"+img_names[img_index])
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
        
        return dataset, shape
    
    if task=='B2':
        
        label=pd.read_csv(path+"/labels.csv",sep='	', index_col=0)
        img_names=list(label['file_name'])
        color=np.array(list(label['eye_color']))
        
        count=0
        start=time.time()
        
        dataset=np.zeros((len(img_names),50,200,4))
        
        print('loading faces and segmenting RoIs...')
        for img_index in range(len(img_names)):
            ProcessBar(count,len(img_names),start)
            count+=1
            img = mpimg.imread(path+"/img/"+img_names[img_index])
            img_colorSeg=img[240:290,150:350,:]
            norm_color = img_colorSeg/255.
            dataset[img_index,:] = norm_color
        
        return dataset, color
        
class model_evaluation():
        
    def metrics(self, pred, ytrue, multiclass, report=False, report_prefix='Test'):
        self.pred = pred
        self.ytrue = ytrue
        self.multiclass = multiclass        
        if self.multiclass:
            self.acc = accuracy_score(self.ytrue, self.pred)
            self.prec = precision_score(self.ytrue, self.pred,average='macro')
            self.recall = recall_score(self.ytrue, self.pred,average='macro')
            self.f1score = f1_score(self.ytrue, self.pred,average='macro')
        else:
            self.acc = accuracy_score(self.ytrue, self.pred)
            self.prec = precision_score(self.ytrue, self.pred)
            self.recall = recall_score(self.ytrue, self.pred)
            self.f1score = f1_score(self.ytrue, self.pred)
        if report:
            print(report_prefix+" Acc: %.4f" %(self.acc))
            print(report_prefix+" Precision: %.4f" %(self.prec))
            print(report_prefix+" recall: %.4f" %(self.recall))
            print(report_prefix+" f1-score: %.4f" %(self.f1score))
            
        return self.acc,self.prec,self.recall,self.f1score
    
    def conf_matrix(self, fig_name="Confision Matrix", axis_rename=None):
        confMat=confusion_matrix(self.ytrue, self.pred)
        confMat=pd.DataFrame(confMat)
        if axis_rename==None:
            None
        else:
            confMat=confMat.rename(index=axis_rename,columns=axis_rename)
        plt.figure(num=fig_name, facecolor='lightgray')
        plt.title(fig_name, fontsize=20)
        ax=sns.heatmap(confMat, fmt='d',cmap='Greys',annot=True)
        ax.set_xlabel('Predicted Class', fontsize=14)
        ax.set_ylabel('True Class', fontsize=14)
        plt.show()        

class model():
    
    def __init__(self, task):
        self.task=task
    
    def train(self,xtr,ytr):
        if self.task=='A1':
            
            self.std = StandardScaler()
            xtr_std = self.std.fit_transform(xtr)
            
            self.classifier=svm.SVC(C=0.9,kernel='rbf')              
            print(str(self.classifier)+" is training...")
            self.classifier.fit(xtr_std,ytr)
            
        elif self.task=='A2':
            
            self.std = StandardScaler()
            xtr_std = self.std.fit_transform(xtr)
            
            self.classifier=BaggingClassifier(n_estimators=500,max_features=12,max_samples=0.3)            
            print(str(self.classifier)+" is training...")            
            self.classifier.fit(xtr_std,ytr)
            
        elif self.task=='B1':
            self.classifier=KNeighborsClassifier(n_neighbors=8)   
            print(str(self.classifier)+" is training...")
            self.classifier.fit(xtr,ytr)
            
        elif self.task=='B2':
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
            ytr = to_categorical(ytr, num_classes=5)
            self.classifier = Sequential()
            self.classifier.add(Conv2D(32, (2,8), input_shape = (50,200,4), activation = 'relu'))
            self.classifier.add(BatchNormalization())
            self.classifier.add(MaxPooling2D(pool_size=(2,4)))
        
            self.classifier.add(Conv2D(64, (3,6), activation = 'relu'))
            self.classifier.add(BatchNormalization()) 
            self.classifier.add(MaxPooling2D(pool_size=(2,4)))
        
            self.classifier.add(Conv2D(64, (3,3), activation = 'relu'))
            self.classifier.add(BatchNormalization())
            self.classifier.add(MaxPooling2D(pool_size=(3,3)))
            
            self.classifier.add(Flatten()) 
            
            self.classifier.add(Dense(units = 64, activation = 'relu'))
            self.classifier.add(BatchNormalization())
            self.classifier.add(Dropout(0.4))
        
            self.classifier.add(Dense(units = 32, activation = 'relu'))
            self.classifier.add(BatchNormalization())
            self.classifier.add(Dropout(0.3))
        
            self.classifier.add(Dense(units = 5, activation = 'softmax')) 
            
            print("CNN is training...")
            print(self.classifier.summary())
            
            self.classifier.compile(optimizer = 'adam' ,
            loss = 'categorical_crossentropy', 
            metrics = ['accuracy'])
    
            def scheduler(epoch):
                if epoch % 15 == 0 and epoch != 0:
                    lr = K.get_value(self.classifier.optimizer.lr)
                    K.set_value(self.classifier.optimizer.lr, lr * 0.1)
                    print("lr changed to {}".format(lr * 0.1))
                return K.get_value(self.classifier.optimizer.lr)
            
            reduce_lr = LearningRateScheduler(scheduler)
            early_stopping=EarlyStopping(monitor='val_accuracy', min_delta=0,
                                      patience=7, verbose=0, mode='auto',
                                      baseline=None, restore_best_weights=True)
            self.classifier.fit(x=xtr, y=xtr,
                           epochs=70,batch_size=32,
                           callbacks=[reduce_lr,early_stopping],
                           verbose=2) 
        else:
            raise ValueError("task param. must be 'A1', 'A2', 'B1', 'B2'")
        return self.classifier
    
    def load_pretrained_model(self):
        from keras.models import load_model
        self.classifier = load_model('CNN.h5')
        print("Pre-trained CNN is loaded.")
        return self.classifier
    
    def predict(self,xte):
        if self.task == 'B1':
            pred = self.classifier.predict(xte)
            return pred
        elif self.task == 'B2':
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
            pred = np.argmax(self.classifier.predict(xte),axis=1)
            return pred
        elif self.task == 'A1':
            pred = self.classifier.predict(self.std.transform(xte))
            return pred
        elif self.task == 'A2':
            pred = self.classifier.predict(self.std.transform(xte))
            return pred            

        
    
    
    
    
    
    
    
    
    
    
    