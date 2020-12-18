# AMLS_assignment20_21_20041534

## Overview

Face is one of the most crucial biometrics to not only distinguish different individuals, but also recognize people’s age, gender and even emotions, etc. There have been numerous successfully commercialized applications in industries, academia and our daily life. Meanwhile, driven by increasing demands from entertainment industry and wide spreading of cartoon characters, cartoon face recognition also demonstrates its application potentials. This report mainly investigates in four tasks: gender recognition, smiling detection for real human’s face and face shape recognition, eye’s color classification for cartoon face. Some advanced feature extraction approaches and machine-learning-based classifiers are compared and used to tackle these tasks. The experiments show the implemented models achieve promising performance in every task.

This project mainly studies and compares various machine learning and neural network models in some classical applications of both real human and cartoon face attributes analysis. The first two tasks are based on a celebrity image (real-human) dataset ([CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)) with 5000 single portraits to implement gender recognition and smiling detection, while an published cartoon face dataset([CartoonSet10k](https://google.github.io/cartoonset/)) with single 10000 portraits is investigated on the face shape recognition and eye’s color classification tasks. Commonly, all of the four tasks are classification problem, while the first two are binary classification and the remaining ones are multi-class classification.

#### Model Summary

| Task | Model         | Features                   | Val Acc | Test Acc |
|:----:|---------------|----------------------------|---------|----------|
|  A1  | SVM           | Face Embedding             | 0.9828  | 0.9938   |
|  A2  | Bagging       | Face Landmarks             | 0.8845  | 0.8804   |
|  B1  | KNN           | Canny Edges                | 0.9999  | 0.9997   |
|  B2  | CNN           | Region of Interest         | 0.8509  | 0.8460   |

## Framework

#### 1) Task A1: Gender Recognition

SVM is a popular machine learning classifier which has promising performance in both linear and non-linear classification problems. To tackle this binary classification problem, Support Vector Machine (SVM) is chosen because of not only its relatively higher validation accuracy compared to others but also its fast training and predicting process.  Furthermore, an observable challenge is that faces can vary significantly even for a same person with different head pose and facial expression. Recognizing common attributes such as gender, age and race require general statistical measurements, e.g. the average nose length, distance between eyes, face shape, etc. Motivated by the face embedding technique to tackle face recognition and clustering problem in this [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html), a pre-trained CNN for face embedding is applied to obtain the 128 face measurements in Task A1, since gender attribute relies on more general and statistical face measurements.

It achieves a 5-fold cross validation accuracy(average) of 98.28%, a testing accuracy of 99.38%. The confusion matrix below shows some details.
<div align=center><img src=https://github.com/JianqiaoMao/AMLS_assignment20_21/blob/main/graphs/Confusion_MatrixA1.png width=400 /></div>

#### 2) Task A2: Smiling Detection

Bagging model is one of ensemble methods which involve many weak base learners to improve performance and overcome overfitting. For smiling detection, RF predicts data class by majority voting of a set of DTs based on the extracted features of estimated face landmarks. A proposed face landmark estimator in this [paper](https://ieeexplore.ieee.org/document/6909637) is implemented to represent not only face shape but also some key facial attributes, which discriminatively describes difference between faces. Intuitively, the shape of lip, eye, chin and eyebrow can efficiently reflect the potential emotions of a person. In particular, a vector of 92 landmark points are extracted from each face, which is composed of 24 points from lip, 12 points from eyes and 10 points from eyebrows.

It achieves a 5-fold cross validation accuracy(average) of 88.45%, a testing accuracy of 88.04%. The confusion matrix below shows some details.
<div align=center><img src=https://github.com/JianqiaoMao/AMLS_assignment20_21/blob/main/graphs/Confusion_MatrixA2.png width=400 /></div>

#### 3) Task B1: Face Shape Recognition

To recognize the given five different types of face shape, K-Nearest Neighbour (KNN) model is applied. KNN is a simple lazy learning model without explicit parameter learning process based on distance between data points. In particular, KNN is believed to have a promising performance in multi-classification problem with balanced sample number in every classes. For Task B1, edges in the cartoon image are extracted by applying [Canny edge detection](https://ieeexplore.ieee.org/document/4767851) as the features to better describe the face shape.

It achieves a 5-fold cross validation accuracy(average) of 99.99%, a testing accuracy of 99.97%. The confusion matrix below shows some details.
<div align=center><img src=https://github.com/JianqiaoMao/AMLS_assignment20_21/blob/main/graphs/Confusion_MatrixB1.png width=400 /></div>

#### 4) Task B2: Eye's Color Classification

Convolutional Neural Network (CNN) is proved to have human-like performance in various computer vision tasks. The convolutional layers, firstly, reduces trainable parameters by weights sharing, secondly, extracts local features of image by convolution operation. Then, fully connected layers perform as classifier based on the extracted feature maps. To classify eye’s color of cartoon characters, the segmented eye’s area (RoI) with a shape of 50-pixel height, 200-pixel width and 4 color channels is used as CNN input.

It achieves a 5-fold cross validation accuracy(average) of 85.09%, a testing accuracy of 84.60%. The confusion matrix below shows some details.
<div align=center><img src=https://github.com/JianqiaoMao/AMLS_assignment20_21/blob/main/graphs/Confusion_MatrixB2.png width=400 /></div>

## File Description

#### Base Directory

1) The file folder **A1**, **A2**, **B1**, **B2** contain code fles for each task. Details is described later.

2) The file folder **Datasets** is set to empty for copy-paste usage by reviwers.

3) The file folder **graphs** contains some png files for demonstration in this **readme** file

4) The .py file **AMLS.py** is the packaged module that should be imported to run **main.py**

5) The .py file **main.py** can be excuted to run the project.

6) The .h5 file **CNN.h5** is the pre-trained CNN model applied into Task B2.

#### File Folder A1

1) **A1_model_selection.py** is run to select traditional machine learning models and tune hyperparameters by grid search and cross validation, including KNN, SVM, Bagging, Boosting.

2) **A1_CNN.py** is run to tune hyperparameters of the CNN and evaluate it by cross validation.

3) **A1_SelectedModel_SVM.py** is run to check the selected SVM performance on test set.

#### File Folder A2

1) **A2_model_selection.py** is run to select traditional machine learning models and tune hyperparameters by grid search and cross validation, including KNN, SVM, Bagging, Boosting.

2) **A2_simple_CNN.py** is run to tune hyperparameters of a relatively simpler CNN and evaluate it by cross validation.

2) **A2_complex_CNN.py** is run to tune hyperparameters of a relatively more complex CNN and evaluate it by cross validation.

4) **A2_SelectedModel_bagging.py** is run to check selected bagging method performance on test set.

#### File Folder B1

1) **B1_model_selection.py** is run to select traditional machine learning models and tune hyperparameters by grid search and cross validation, including KNN, SVM, Bagging, Boosting.

2) **B1_CNN.py** is run to tune hyperparameters of the CNN and evaluate it by cross validation.

3) **B1_SelectedModel_KNN.py** is run to check the selected KNN performance on test set.

#### File Folder B2

1) **B1_CNN_simple.py** is run to tune hyperparameters of a relatively simpler CNN and evaluate it by cross validation.

2) **B2_CNN_complicated.py** is run to tune hyperparameters of a relatively more complex CNN and evaluate it by cross validation.

3) **B2_SelectedModel_CNN.py** is run to check the selected CNN performance on test set.

## Dependent Environment and Tips

#### Dependent Environment

The whole project is developed in Python3.6. Please note that using other Python versions may lead to unknown errors. Required libraries are shown below, where the recommended package versions are also demonstrated:

  * numpy 1.19.1
  * pandas 1.1.3
  * face_recognition 1.3.0
  * dlib 19.21.0
  * opencv-python 4.4.0.46
  * seaborn 0.11.0
  * matplotlib 3.3.2
  * scikit-learn 0.23.2
  * keras 2.4.3
  * tensorflow-gpu 2.3.1 / Alternative: tensorflow (latest version)
  
Note that the CNN models are built by tensorflow-gpu version, while it is uncertain for its compatibility in the basic tensorflow module. Conflict may happen if you have both of the two package, since base tensorflow is the default to be imported. Some other dependent libraries may be required to apply **face_recognition** module, if meet errors, please check [here](https://github.com/ageitgey/face_recognition).

#### Tips

The file reading directory (for dataset loading) is tested on Windows10 (x86) using Spyder as IDE, while uncertainty can be expected for running on OS or Linux or other IDEs. If errors encountered, please modify the **pathA**, **pathB**, **pathA_addition**, **pathB_addition** variables in **main.py** at line 8-11. The data reading works on relative directory, try to excute **main.py** in the directory where it locates.
