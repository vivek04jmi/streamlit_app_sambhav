# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:31:33 2021

@author: Divya.Upadhyay
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve

# Validation
from sklearn.pipeline import Pipeline, make_pipeline

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer, LabelEncoder

# Models
from sklearn.naive_bayes import GaussianNB

# Ensembles
from sklearn.ensemble import RandomForestClassifier

#
#                Split Data to Training and Validation set                     #
#                                                                              #
################################################################################
def read_in_and_split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train=data.drop(target, axis=1)
    X_test=data.drop(target, axis=1)
    y_train=data[target]
    y_test=data[target]
    return X_train, X_test, y_train, y_test
################################################################################
#                                                                              #
#                        Spot-Check Algorithms                                 #
#                                                                              #
################################################################################
def GetModel():
    Models = []
    Models.append(('NB'   , GaussianNB()))
    return Models

def ensemblemodels():
    ensembles = []
    ensembles.append(('RF'   , RandomForestClassifier()))
    return ensembles
################################################################################
#                                                                              #
#                 Spot-Check Normalized Models                                 #
#                                                                              #
################################################################################
def NormalizedModel(nameOfScaler):
    
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()
    elif nameOfScaler == 'normalizer':
        scaler = Normalizer()
    elif nameOfScaler == 'binarizer':
        scaler = Binarizer()

    pipelines = []
    pipelines.append((nameOfScaler+'GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier())])  ))
    pipelines.append((nameOfScaler+'RF'  , Pipeline([('Scaler', scaler),('RF'  , RandomForestClassifier())])  ))
    
    return pipelines
################################################################################
#                                                                              #
#                           Train Model                                        #
#                                                                              #
################################################################################
def fit_model(X_train, y_train,models):
    # Test options and evaluation metric
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    return names, results
################################################################################
#                                                                              #
#                          Save Trained Model                                  #
#                                                                              #
################################################################################
def save_model(model,filename):
    pickle.dump(model, open(filename, 'wb'))
################################################################################


# Load Dataset
df = pd.read_csv('Crop_recommendation.csv')

# Remove Outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Split Data to Training and Validation set
target ='label'
X_train, X_test, y_train, y_test = read_in_and_split_data(df, target)

# Train model
pipeline = make_pipeline(StandardScaler(),  GaussianNB())
model = pipeline.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test,y_pred)
#classification_metrics(pipeline, conf_matrix)

# save model
#save_model(model, 'model.pkl')

import streamlit as st 
import pandas as pd
# import numpy as np
# import os
import pickle
#import warnings


#st.beta_set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")
def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:BLUE;text-align:left;"> Crop Recommendation App </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("""
             This app recommends the best crop out of 22 crops to be sown during 
             the sowing season based on factors like soil condition 
             (N, P, K content, soil pH value), temperature and  precipitation""")
    N = st.number_input("Nitrogen", 1,10000)
    P = st.number_input("Phosporus", 1,10000)
    K = st.number_input("Potassium", 1,10000)
    temp = st.number_input("Temperature",0.0,100000.0)
    humidity = st.number_input("Humidity in %", 0.0,100000.0)
    ph = st.number_input("Ph", 0.0,100000.0)
    rainfall = st.number_input("Rainfall in mm",0.0,100000.0)

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1,-1)
    
    if st.button('Predict'):
       model = pipeline.fit(X_train, y_train)
       y_pred = model.predict(X_test)
       loaded_model=model
#           loaded_model = load_model('model.pkl')
       prediction = loaded_model.predict(single_pred)
       
       st.write('''
		    ## Results 🔍 
		    ''')
       st.success(f"{prediction.item().title()} are recommended for your farm.")
       
#     st.warning("Note: This A.I application is for educational/demo purposes only and cannot be relied upon. Check the source code [here](https://github.com/gabbygab1233/Crop-Recommendation)")
#     hide_menu_style = """
#     <style>
#     #MainMenu {visibility: hidden;}
#     </style>
#     """

# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
 	main()