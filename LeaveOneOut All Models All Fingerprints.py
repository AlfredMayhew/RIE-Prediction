#This is the code written to test RIE predictions from a wide range of 
#model-fingerprint combinations. Bayesian Ridge regression was found to 
#give the best predictions with our dataset. A separate script has been 
#provided where only Bayesian Ridge Regression is used, which will require
#less time to run. 

#imports and turn off warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
import umansysprop.groups
import pybel

# Set max row display
pd.set_option('display.max_row', 1000)

# Set max column width to 50
pd.set_option('display.max_columns', 50)

#define scaler
scaler=RobustScaler()

#create fingerprints from SMILES
stringsmiles=open("RIE-Data/Selected SMILES Structures.txt")
smiles0=stringsmiles.readlines()
SMILES=[]
for i in smiles0:
    SMILES.append(i.strip())

#create dataframe of all RIE Values
dfRIE=pd.read_csv("RIE-Data/Selected SMILES and logRIEs.csv",index_col=0)

#List of all the fingerprints to be tested (and a list of their names 
#for the tables)
available_fingerprints=[umansysprop.groups.composition,
                        umansysprop.groups.stein_and_brown,
                        umansysprop.groups.nannoolal_primary,
                        umansysprop.groups.nannoolal_secondary,
                        umansysprop.groups.nannoolal_interactions,
                        umansysprop.groups.evaporation,
                        umansysprop.groups.girolami,
                        umansysprop.groups.schroeder,umansysprop.groups.le_bas,
                        umansysprop.groups.unifac,umansysprop.groups.aiomfac]

fingerprint_names=["Composition","Stein and Brown","Nannoolal Primary",
                   "Nannoolal Secondary","Nannoolal Interactions","Evaporation"
                   ,"Girolami","Schroeder","Le Bas","unifac","aiomfac"]


#Define each model to be tested, adjust model parameters, and put into list.
linearRegressionModel=linear_model.LinearRegression()
bayesianModel=linear_model.BayesianRidge(alpha_1=10)
decisionTreeModel=DecisionTreeRegressor(min_samples_split=3)
mlpModel=MLPRegressor(hidden_layer_sizes=40,solver="lbfgs",
                      max_iter=500,tol=0.3)
passiveAggressiveModel=linear_model.PassiveAggressiveRegressor(max_iter=1000, 
                                                               tol=1e-3)
randomForestModel=RandomForestRegressor(n_estimators=10,max_depth=10,
                                        min_samples_split=5,min_samples_leaf=4,
                                        min_weight_fraction_leaf=0.04,
                                        max_leaf_nodes=20)
sgdModel=linear_model.SGDRegressor(max_iter=100, tol=2)
svrModel=SVR(kernel="poly", gamma=5)

models=[linearRegressionModel,bayesianModel,decisionTreeModel,mlpModel,
        passiveAggressiveModel,randomForestModel,sgdModel,svrModel]

#Need a separate list of model names as printing the models also gives all 
#the settings for each one
model_names=["Linear Regression","Bayesian Ridge","Decision Tree","MLP",
             "Passive Aggressive","Random Forest","SGD","SVR"]

#Define y variable for regression (same for all fingerprints and models)
y=np.array(dfRIE.iloc[:,0:1].reset_index(drop=True))

#Create dictionary of dataframes to populate with predicted values for each 
#compound as different fingerprints using each model
predicted_vals={}
for m in model_names:
    predicted_vals[m]=pd.DataFrame(index=SMILES, columns=fingerprint_names)

#Performing LOOCV for each model, so iterate through list of models
fingerprint_count=0
for f in available_fingerprints:
    #More easily refer to which fingerprint is being used
    fingerprint=fingerprint_names[fingerprint_count] 
    keys = {}
    for s in SMILES:
        SMILES_object=pybel.readstring('smi',s)
        keys[s]=f(SMILES_object)
    
    #Make dataframe of keys/fingerprints, scaling with sklearn
    keys_scaled=scaler.fit_transform(pd.DataFrame(keys).transpose().values)
    dfParam=pd.DataFrame(keys_scaled,index=SMILES)

    #Define x variable for regression
    x=np.array(dfParam)
    
    model_count=0
    for i in models:
        model_name=model_names[model_count]
        for train_index, test_index in LeaveOneOut().split(x, y=y):
           #define training and test sets from the LOO split
           x_train=x[train_index]
           x_test=x[test_index]
           y_train=y[train_index]
           y_test=y[test_index]
           
           #Train model and then predict value for each test compound
           model = i
           model.fit(X= x_train, y = y_train) 
           y_pred = model.predict(x_test)
           
           #Input predicted value to the blank dataframe (if/else required as 
           #output from Linear Regression is different from the other models)
           if model_count==0:
               predicted_vals[model_name].loc[SMILES[test_index[0]],
                              fingerprint]=y_pred[0][0]
           else:
               predicted_vals[model_name].loc[SMILES[test_index[0]],
                              fingerprint]=y_pred[0]
        model_count=model_count+1
    fingerprint_count=fingerprint_count+1


#Define type of score and error being used
score=metrics.r2_score
error=metrics.mean_squared_error

#Create empty dataframe to populate with scores of each model
modelscores=pd.DataFrame(index=model_names,columns=fingerprint_names)
modelerror=pd.DataFrame(index=model_names,columns=fingerprint_names)

#Define a dataframe of experimental values
ytests=dfRIE.loc[:,"logRIE"].tolist()

#Iterate through each model with each fingerprint, calculating score and error 
#for each and inputing into dataframe
for m in model_names:
    for f in fingerprint_names:
        ypreds=predicted_vals[m].loc[:,f]
        
        modelscores.loc[m,f]= score(y_true=ytests, y_pred=ypreds)
        modelerror.loc[m,f]= error(ytests, ypreds)
    

#Print scores for each model       
print("Model Scores")
print(modelscores)
print()
print("Model Error")
print(modelerror)
print("-"*50)

#Find model/fingerprint combo that gives highest score, and print result
print("Best Model-Fingerprint Combo (Based on Model Scores): ")
print("Fingerprint: ",modelscores.max(axis=0).idxmax())
print("Model Type: ",modelscores.max(axis=1).idxmax())
print("Score: ",modelscores.loc[modelscores.max(axis=1).idxmax(),
                                modelscores.max(axis=0).idxmax()])

#plot true vs predicted graph for the best model
plt.plot(dfRIE.loc[:,"logRIE"], 
         predicted_vals[modelscores.max(axis=1).idxmax()].loc[:,
         modelscores.max(axis=0).idxmax()],"o")
plt.plot([-3,5],[-3,5])
plt.xlabel("Measured logRIE")
plt.ylabel("Predicted logRIE")
plt.xlim(-2,4)
plt.ylim(-2,4)
plt.pause(1E-100)
plt.show()

print("-"*50)


