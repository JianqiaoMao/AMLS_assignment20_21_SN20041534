#%%---------------------Configuration---------------------
from sklearn.model_selection import train_test_split
import AMLS
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

pathA="./Datasets/celeba"
pathB="./Datasets/cartoon_set"
pathA_addition = "./Datasets/celeba_test"
pathB_addition = "./Datasets/cartoon_set_test"

#%%---------------------Task A1---------------------
print("========================Task A1========================")
## Loading and spliting dataset ##
print("Loading and spliting dataset...")
datasetA1,labelA1 = AMLS.form_dataset(pathA,'A1')
X_train_A1,X_test_A1,y_train_A1,y_test_A1=train_test_split(datasetA1,labelA1,
                                               test_size=0.1,random_state=3) 
## Model-A1 training ##
print("\nModel-A1 training...")
modelA1 = AMLS.model('A1')
modelA1.train(X_train_A1,y_train_A1)  

## Model-A1 evaluating ##
print("Model-A1 evaluating...")
evaluatorA1 = AMLS.model_evaluation()
# Training metrics #
pred_train_A1 = modelA1.predict(X_train_A1)
metrics_train_A1 = evaluatorA1.metrics(pred_train_A1, y_train_A1, multiclass=False)
# Testing metrics #
pred_test_A1 = modelA1.predict(X_test_A1)
metrics_test_A1 = evaluatorA1.metrics(pred_test_A1, y_test_A1, 
                                      multiclass=False, report=True, 
                                      report_prefix="A1: Self-split Test")
evaluatorA1.conf_matrix(fig_name="A1 Confision Matrix(self built test set)",axis_rename={0:'Female',1:'Male'})

print("++++++++++++++++++++++++++++++++++++++++++++++++")
## Loading additional dataset ##
print("\nLoading and spliting additional dataset...")
X_addTest_A1, y_addTest_A1 = AMLS.form_dataset(pathA_addition,'A1')

## Test on additional dataset ##
print("\nTesting on additional dataset...")
pred_addTest_A1 = modelA1.predict(X_addTest_A1)
metrics_addTest_A1 = evaluatorA1.metrics(pred_addTest_A1, y_addTest_A1,
                                         multiclass=False, report=True, 
                                         report_prefix="A1: Additional Test")
evaluatorA1.conf_matrix(fig_name="A1 Confision Matrix(additional test set)",axis_rename={0:'Female',1:'Male'})

#%%---------------------Task A2---------------------
print("========================Task A2========================")
## Loading and spliting dataset ##
print("Loading and spliting dataset...")
datasetA2,labelA2 = AMLS.form_dataset(pathA,'A2')
X_train_A2,X_test_A2,y_train_A2,y_test_A2=train_test_split(datasetA2,labelA2,
                                               test_size=0.1,random_state=3) 
## Model-A2 training ##
print("\nModel-A2 training...")
modelA2 = AMLS.model('A2')
modelA2.train(X_train_A2,y_train_A2)  

## Model-A2 evaluating ##
print("Model-A2 evaluating...")
evaluatorA2 = AMLS.model_evaluation()
# Training metrics #
pred_train_A2 = modelA2.predict(X_train_A2)
metrics_train_A2 = evaluatorA2.metrics(pred_train_A2, y_train_A2, multiclass=False)
# Testing metrics #
pred_test_A2 = modelA2.predict(X_test_A2)
metrics_test_A2 = evaluatorA2.metrics(pred_test_A2, y_test_A2, 
                                      multiclass=False, report=True, 
                                      report_prefix="A2: Self-split Test")
evaluatorA2.conf_matrix(fig_name="A2 Confision Matrix(self built test set)",axis_rename={0:'Unsmiling',1:'Smiling'})

print("++++++++++++++++++++++++++++++++++++++++++++++++")
## Loading additional dataset ##
print("\nLoading and spliting additional dataset...")
X_addTest_A2, y_addTest_A2 = AMLS.form_dataset(pathA_addition,'A2')

## Test on additional dataset ##
print("\nTesting on additional dataset...")
pred_addTest_A2 = modelA2.predict(X_addTest_A2)
metrics_addTest_A2 = evaluatorA2.metrics(pred_addTest_A2, y_addTest_A2, 
                                         multiclass=False, report=True, 
                                         report_prefix="A2: Additional Test")
evaluatorA2.conf_matrix(fig_name="A2 Confision Matrix(additional test set)",axis_rename={0:'Unsmiling',1:'Smiling'})

#%%---------------------Task B1---------------------
print("========================Task B1========================")
## Loading and spliting dataset ##
print("Loading and spliting dataset...")
datasetB1,labelB1 = AMLS.form_dataset(pathB,'B1')
X_train_B1,X_test_B1,y_train_B1,y_test_B1=train_test_split(datasetB1,labelB1,
                                               test_size=0.1,random_state=3)
## Model-B1 training ## 
print("\nModel-B1 training...")
modelB1 = AMLS.model('B1')
modelB1.train(X_train_B1,y_train_B1)  

## Model-B1 evaluating ##
print("Model-B1 evaluating...")
evaluatorB1 = AMLS.model_evaluation()

pred_train_B1 = modelB1.predict(X_train_B1)
metrics_train_B1 = evaluatorB1.metrics(pred_train_B1, y_train_B1, multiclass=True)

pred_test_B1 = modelB1.predict(X_test_B1)
metrics_test_B1 = evaluatorB1.metrics(pred_test_B1, y_test_B1, 
                                      multiclass=True, report=True,
                                      report_prefix = "B1:  Self-split Test")
evaluatorB1.conf_matrix(fig_name="B1 Confision Matrix(self built test set)")

print("++++++++++++++++++++++++++++++++++++++++++++++++")
## Loading additional dataset ##
print("\nLoading and spliting additional dataset...")
X_addTest_B1, y_addTest_B1 = AMLS.form_dataset(pathB_addition,'B1')

## Test on additional dataset ##
print("\nTesting on additional dataset...")
pred_addTest_B1= modelB1.predict(X_addTest_B1)
metrics_addTest_B1 = evaluatorB1.metrics(pred_addTest_B1, y_addTest_B1, 
                                         multiclass=True, report=True, 
                                         report_prefix="B1: Additional Test")
evaluatorB1.conf_matrix(fig_name="B1 Confision Matrix(additional test set)")

#%%---------------------Task B2---------------------
print("========================Task B2========================")
## Loading and spliting dataset ##
print("Loading and spliting dataset...")
datasetB2,labelB2 = AMLS.form_dataset(pathB,'B2')
X_train_B2,X_test_B2,y_train_B2,y_test_B2=train_test_split(datasetB2,labelB2,
                                               test_size=0.1,random_state=3) 

## Loading pre-trained CNN ##
print("\nModel-B2 loading...")
modelB2 = AMLS.model('B2')
modelB2.load_pretrained_model()

## Model-B2 evaluating ##
print("Model-B2 evaluating...")
evaluatorB2 = AMLS.model_evaluation()

pred_train_B2 = modelB2.predict(X_train_B2)
metrics_train_B2 = evaluatorB2.metrics(pred_train_B2, y_train_B2, multiclass=True)

pred_test_B2 = modelB2.predict(X_test_B2)
metrics_test_B2 = evaluatorB2.metrics(pred_test_B2, y_test_B2, 
                                      multiclass=True, report=True,
                                      report_prefix="B2: Self-split Test")
evaluatorB2.conf_matrix(fig_name="B2 Confision Matrix(self built test set)")

print("++++++++++++++++++++++++++++++++++++++++++++++++")
## Loading additional dataset ##
print("\nLoading and spliting additional dataset...")
X_addTest_B2, y_addTest_B2 = AMLS.form_dataset(pathB_addition,'B2')

## Test on additional dataset ##
print("\nTesting on additional dataset...")
pred_addTest_B2= modelB2.predict(X_addTest_B2)
metrics_addTest_B2 = evaluatorB2.metrics(pred_addTest_B2, y_addTest_B2, 
                                         multiclass=True, report=True, 
                                         report_prefix="B2: Additional Test")
evaluatorB2.conf_matrix(fig_name="B2 Confision Matrix(additional test set)")
