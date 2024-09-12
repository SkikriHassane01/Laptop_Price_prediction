import streamlit as st 
import joblib
import pandas as pd
import os 
import numpy as np
import regex as re
from sklearn.base import BaseEstimator, TransformerMixin
# at first we need to add the feature creation to the start of our pipeline so let's do it:
# here is all the preprocessing that we did above:


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Extract resolution and ScreenType
        X['resolution'] = X['ScreenResolution'].str.extract(r'(\d+x\d+)')
        X['ScreenType'] = X['ScreenResolution'].replace(r'(\d+x\d+)', '', regex=True)
        X['ScreenType'] = X['ScreenType'].replace(r'^\s*$', np.nan, regex=True)
        
        # Extract TouchScreen
        X['TouchScreen'] = X['ScreenType'].str.extract(r'(Touchscreen)').notna().astype(int)
        X['ScreenType'] = X['ScreenType'].str.replace(r'(\/\sTouchscreen)', '', regex=True)
        X['ScreenType'] = X['ScreenType'].replace(np.nan, X['ScreenType'].mode()[0])

        # Drop ScreenResolution
        X = X.drop('ScreenResolution', axis=1)
        
        # Extract CpuFrequency
        X['CpuFrequency'] = X['Cpu'].str.extract(r'(\d+\.?\d*GHz)').replace('GHz', '', regex=True).astype(float)
        X['Cpu'] = X['Cpu'].str.replace(r'\d+\.?\d*GHz', '', regex=True)
        
        # Convert Ram
        X['Ram'] = X['Ram'].str.replace('GB', '').astype(int)
        
        # Process Memory
        X['Memory'] = X['Memory'].str.replace('1.0TB', '1000GB').str.replace('1TB', '1000GB').str.replace('2TB', '2000GB').str.replace('GB', '')
        X['Memory'] = X['Memory'].str.replace(' ', '')
        
        # Extract storageDisk1 and storageDisk2
        X['storageDisk1'] = X['Memory'].str.extract(r'(^\d+)').astype(int)
        X['storageDisk2'] = X['Memory'].str.extract(r'(\+\d+)')
        X['storageDisk2'] = X['storageDisk2'].fillna('0').str.replace('+', '').astype(int)

        # Extract TypeDisk1 and TypeDisk2
        TypeDisk1 = []
        TypeDisk2 = []
        for i in X['Memory']:
            if len(re.findall(r'\+', i)) == 1:
                allTypes = re.findall(r'([A-z]+)', i)
                TypeDisk1.append(allTypes[0])
                TypeDisk2.append(allTypes[1])
            else:
                allTypes = re.findall(r'([A-z]+)', i)
                TypeDisk1.append(allTypes[0])
                TypeDisk2.append(np.nan)
        
        X['TypeDisk1'] = TypeDisk1
        X['TypeDisk2'] = TypeDisk2
        X['TypeDisk2'] = X['TypeDisk2'].fillna('NaN')
        
        # Drop Memory column
        X = X.drop(columns=['Memory'], axis=1)
        
        # Convert Weight to numeric
        X['Weight'] = X['Weight'].str.replace('kg', '').astype(float)
        
        return X
    
# load the pipeline 
pipeline = joblib.load('pipeline.joblib')

st.title('Laptop Price Predictor Application 💻')


user_input = {
    "Company": st.text_input("Company",placeholder="Apple , HP ,  Acer ... "),
    "Product": st.text_input("Product",placeholder=" MacBook Pro ,  Macbook Air ..."),
    "TypeName": st.selectbox("Type", options=['Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible','Workstation']),
    "Inches": st.number_input("Screen Size (Inches)", min_value=10.0, max_value=19.0),
    "ScreenResolution": st.text_input("Screen Resolution",placeholder="IPS Panel Retina Display 2560x1600 ..."),
    "Cpu": st.text_input("CPU",placeholder="Intel Core i5 2.3GHz ..."),
    "Ram": st.text_input("RAM",placeholder="8GB 16GB ..."),
    "Memory": st.text_input("Memory",placeholder="128GB SSD, 128GB Flash Storage"),
    "Gpu": st.text_input("GPU", placeholder="Intel Iris Plus Graphics 640"),
    "OpSys": st.selectbox("Operating System", options=['macOS', 'No OS', 'Windows 10', 'Mac OS X', 'Linux', 'Android','Windows 10 S', 'Chrome OS', 'Windows 7']),
    "Weight": st.text_input("Weight",placeholder="1.34kg"),
}

file_txt = "file.txt"

if st.button('Predict Price'):
    input_data = pd.DataFrame([user_input])

    if not os.path.exists(file_txt):
        with open(file_txt, 'w') as f:
            f.write("") 

    with open(file_txt, 'a') as f:
        input_data.to_csv(f, header=False, index=False) 

    predicted_price = pipeline.predict(input_data)

    st.write(f"Predicted Price: {predicted_price[0]:.2f} €")