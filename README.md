# DeepNASH

A Competing Risk Neural Network Model For Forecasting NASH Patient Trajectories on the Liver Transplant Waitlist


## Model description 
DeepNASH has been trained using SRTR data of NASH patients at the time of listing and the DeepHit model structure to predict the monthly risk of dying on the death and receiving a liver transplant. 

## Data requirement for DeepNASH
The processing of the data was done using [data_processing.py](https://github.com/criticalml-uw/DeepNASH/blob/main/data/data_processing.py) where the categeorizaiton of features based on clinical definitions, missing data imputation, and one hot encoding were done. Sample data can be viewed here [sample data](https://github.com/criticalml-uw/DeepNASH/blob/main/data/sample_test_data.csv). The template of input features,[data upload template](https://github.com/criticalml-uw/DeepNASH/blob/main/data/test_data_template.csv), can be viewed and downloaded to be filled and served as input for the model. 

## Using the model 

### Option 1: Use trained model from scripts
The [trained DeepNASH model](https://github.com/criticalml-uw/DeepNASH/tree/main/model/model) is availble for use with the following files [DeepHit_Prediction.py](https://github.com/criticalml-uw/DeepNASH/blob/main/DeepHit_Prediction.py) data in the template format. 

### Option 2: Streamlit Dashboard 
The model has been implemented in Streamlit with functions of downloading data templates, uploading patient data as *.csv* files, and visualizing the risk forecast. 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deepnash.streamlit.app/)

# 

