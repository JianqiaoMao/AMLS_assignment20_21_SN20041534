# AMLS_assignment20_21_20041534

## Overview:

Face is one of the most crucial biometrics to not only distinguish different individuals, but also recognize people’s age, gender and even emotions, etc. There have been numerous successfully commercialized applications in industries, academia and our daily life. Meanwhile, driven by increasing demands from entertainment industry and wide spreading of cartoon characters, cartoon face recognition also demonstrates its application potentials. This report mainly investigates in four tasks: gender recognition, smiling detection for real human’s face and face shape recognition, eye’s color classification for cartoon face. Some advanced feature extraction approaches and machine-learning-based classifiers are compared and used to tackle these tasks. The experiments show the implemented models achieve promising performance in every task.

This project mainly studies and compares various machine learning and neural network models in some classical applications of both real human and cartoon face attributes analysis. The first two tasks are based on a celebrity image (real-human) dataset ([CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)) with 5000 single portraits to implement gender recognition and smiling detection, while an published cartoon face dataset([CartoonSet10k](https://google.github.io/cartoonset/)) with single 10000 portraits is investigated on the face shape recognition and eye’s color classification tasks. Commonly, all of the four tasks are classification problem, while the first two are binary classification and the remaining ones are multi-class classification.

## Framework

### 1) Task A1: Gender Recognition

SVM is a popular machine learning classifier which has promising performance in both linear and non-linear classification problems. To tackle this binary classification problem, Support Vector Machine (SVM) is chosen because of not only its relatively higher validation accuracy compared to others but also its fast training and predicting process.  Furthermore, an observable challenge is that faces can vary significantly even for a same person with different head pose and facial expression. Recognizing common attributes such as gender, age and race require general statistical measurements, e.g. the average nose length, distance between eyes, face shape, etc. Motivated by the face embedding technique to tackle face recognition and clustering problem in this [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html), a pre-trained CNN for face embedding is applied to obtain the 128 face measurements in Task A1, since gender attribute relies on more general and statistical face measurements.

It achieves a 5-fold cross validation accuracy(average) of 98.28%, a testing accuracy of 99.38%. The confusion matrix below shows some details.
<div align=center><img src=https://github.com/JianqiaoMao/AMLS_assignment20_21/blob/main/graphs/Confusion_MatrixA1.png width=400 /></div>

### 2) Task A2: Smiling Detection

Bagging model is one of ensemble methods which involve many weak base learners to improve performance and overcome overfitting. For smiling detection, RF predicts data class by majority voting of a set of DTs based on the extracted features of estimated face landmarks. A proposed face landmark estimator in this [paper](https://ieeexplore.ieee.org/document/6909637) is implemented to represent not only face shape but also some key facial attributes, which discriminatively describes difference between faces. Intuitively, the shape of lip, eye, chin and eyebrow can efficiently reflect the potential emotions of a person. In particular, a vector of 92 landmark points are extracted from each face, which is composed of 24 points from lip, 12 points from eyes and 10 points from eyebrows.

It achieves a 5-fold cross validation accuracy(average) of 88.45%, a testing accuracy of 88.04%. The confusion matrix below shows some details.
<div align=center><img src=https://github.com/JianqiaoMao/AMLS_assignment20_21/blob/main/graphs/Confusion_MatrixA2.png width=400 /></div>

### 3) Task B1: Face Shape Recognition

To recognize the given five different types of face shape, K-Nearest Neighbour (KNN) model is applied. KNN is a simple lazy learning model without explicit parameter learning process based on distance between data points. In particular, KNN is believed to have a promising performance in multi-classification problem with balanced sample number in every classes. For Task B1, edges in the cartoon image are extracted by applying [Canny edge detection](https://ieeexplore.ieee.org/document/4767851) as the features to better describe the face shape.

It achieves a 5-fold cross validation accuracy(average) of 99.99%, a testing accuracy of 99.97%. The confusion matrix below shows some details.
<div align=center><img src=https://github.com/JianqiaoMao/AMLS_assignment20_21/blob/main/graphs/Confusion_MatrixB1.png width=400 /></div>
