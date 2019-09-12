import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn import linear_model
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
stringsmiles=open("Selected SMILES Structures.txt")
smiles0=stringsmiles.readlines()
SMILES=[]
for i in smiles0:
    SMILES.append(i.strip())

#create dataframe of all RIE Values
dfRIE=pd.read_csv("Selected SMILES and logRIEs.csv",index_col=0)

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


#Define each model to be tested, and put into list.
bayesianModel=linear_model.BayesianRidge(alpha_1=10)

#Define y variable for regression (same for all fingerprints and models)
y=np.array(dfRIE.iloc[:,0:1].reset_index(drop=True))

#Create dictionary of dataframes to populate with predicted values for each 
#compound as different fingerprints using each model
predicted_vals=pd.DataFrame(index=SMILES, columns=fingerprint_names)

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
    
    for train_index, test_index in LeaveOneOut().split(x, y=y):
       #define training and test sets from the LOO split
       x_train=x[train_index]
       x_test=x[test_index]
       y_train=y[train_index]
       y_test=y[test_index]
       
       #Train model and then predict value for each test compound
       bayesianModel.fit(X= x_train, y = y_train) 
       y_pred = bayesianModel.predict(x_test)
       
       #Input predicted value to the blank dataframe
       predicted_vals.loc[SMILES[test_index[0]],
                      fingerprint]=y_pred[0]
    fingerprint_count=fingerprint_count+1


#Define type of score and error being used
score=metrics.r2_score
error=metrics.mean_squared_error

#Create empty dataframe to populate with scores of each model
modelscores=pd.DataFrame(index=fingerprint_names,
                         columns=["Model Score","Model Error"])

#Define a dataframe of experimental values
y_exp=dfRIE.loc[:,"logRIE"].tolist()

#Iterate through each model with each fingerprint, calculating score and error 
#for each and inputing into dataframe
for f in fingerprint_names:
    y_pred=predicted_vals.loc[:,f]
    
    modelscores.loc[f,"Model Score"]= score(y_true=y_exp, y_pred=y_pred)
    modelscores.loc[f,"Model Error"]= error(y_exp, y_pred)

#Print scores for each model       
print("Model Scores")
print(modelscores)
print("-"*50)

#Find best Fingerprint
best_fingerprint=modelscores.loc[:,"Model Score"].astype(float).idxmax()

print("Best Fingerprint: ", best_fingerprint)
print("Model Score: ", modelscores.loc[best_fingerprint,"Model Score"])
print("Model Error: ", modelscores.loc[best_fingerprint,"Model Error"])

#plot true vs predicted graph for the best model
plt.plot(dfRIE.loc[:,"logRIE"], 
         predicted_vals.loc[:,best_fingerprint],"o")
plt.plot([-3,5],[-3,5])
plt.xlabel("Measured logRIE")
plt.ylabel("Predicted logRIE")
plt.xlim(-2,4)
plt.ylim(-2,4)
plt.pause(1E-100)
plt.show()

print("-"*50)


