#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title("Accident Analysis and Prediction")

# File path for the dataset
file_path = "D:/2024S-T2 BDM 2203 - Big Data Visualization for Business Communications 01/Final_Project/Vic_Road_Crash_Data.csv"

# Upload the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(file_path)  # Load the dataset from the file path if no file is uploaded

st.write("Data Preview", df.head())

# Preprocessing steps
df_ml = df.copy()

# Encode categorical features
label_encoders = {}
for column in ['ACCIDENT_TYPE_DESC', 'DAY_WEEK_DESC', 'DCA_DESC', 'LIGHT_CONDITION', 'ROAD_GEOMETRY_DESC']:
    le = LabelEncoder()
    df_ml[column] = le.fit_transform(df_ml[column])
    label_encoders[column] = le

# Sidebar for selecting task
st.sidebar.title("Choose Analysis")
task = st.sidebar.selectbox("Select Task", ["Severity Prediction", "Accident Type Prediction", "Accident Hotspot Clustering", "Accident Forecasting", "Anomaly Detection"])

# Define features for Severity Prediction (these will be used for both severity prediction and anomaly detection)
X_severity = df_ml[['ACCIDENT_TYPE_DESC', 'DAY_OF_WEEK', 'LIGHT_CONDITION', 'SPEED_ZONE', 'ROAD_GEOMETRY_DESC']]
y_severity = df_ml['SEVERITY']

if task == "Severity Prediction":
    st.subheader("Severity Prediction Using RandomForest")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_severity, y_severity, test_size=0.3, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Feature importance
    feat_importances = pd.Series(rf_model.feature_importances_, index=['ACCIDENT_TYPE_DESC', 'DAY_OF_WEEK', 'LIGHT_CONDITION', 'SPEED_ZONE', 'ROAD_GEOMETRY_DESC'])
    st.write("Feature Importances", feat_importances.nlargest(5))

    # Visualization
    fig, ax = plt.subplots()
    feat_importances.nlargest(5).plot(kind='barh', color='green', ax=ax)
    ax.set_title('Feature Importance - Severity Prediction')
    st.pyplot(fig)

elif task == "Accident Type Prediction":
    st.subheader("Accident Type Prediction Using RandomForest")
    
    # Define features and target
    X_accident_type = df_ml[['DAY_OF_WEEK', 'LIGHT_CONDITION', 'SPEED_ZONE', 'ROAD_GEOMETRY_DESC']]
    y_accident_type = df_ml['ACCIDENT_TYPE_DESC']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_accident_type, y_accident_type, test_size=0.3, random_state=42)

    # Train the model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    st.write("Accident Type Prediction Model Trained Successfully!")

elif task == "Accident Hotspot Clustering":
    st.subheader("Accident Hotspot Identification Using K-Means Clustering")
    
    # Clustering Models: Accident Hotspot Identification
    df_cluster = df[['NODE_ID', 'SEVERITY']].dropna()
    df_cluster['NODE_ID'] = df_cluster['NODE_ID'].astype(int)

    # Standardize the data
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df_cluster)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df_cluster['Cluster'] = kmeans.fit_predict(X_cluster)

    # Visualize clusters
    pca = PCA(2)
    cluster_pca = pca.fit_transform(X_cluster)
    df_cluster['PCA1'] = cluster_pca[:, 0]
    df_cluster['PCA2'] = cluster_pca[:, 1]

    fig, ax = plt.subplots()
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_cluster, palette='viridis', ax=ax)
    ax.set_title('Accident Hotspot Clusters')
    st.pyplot(fig)

elif task == "Accident Forecasting":
    st.subheader("Accident Occurrence Forecast Using ARIMA")
    
    # Ensure that the 'ACCIDENT_DATE' column is in datetime format
    df['ACCIDENT_DATE'] = pd.to_datetime(df['ACCIDENT_DATE'], dayfirst=True)
    
    # Set 'ACCIDENT_DATE' as the index
    df_timeseries = df.set_index('ACCIDENT_DATE')
    
    # Resample the data by month to get the number of accidents per month
    df_timeseries = df_timeseries.resample('M').size()
    
    # ARIMA model for forecasting
    arima_model = ARIMA(df_timeseries, order=(5, 1, 0))
    arima_fit = arima_model.fit()

    # Forecasting
    forecast = arima_fit.forecast(steps=12)
    fig, ax = plt.subplots()
    ax.plot(df_timeseries, label='Actual')
    ax.plot(forecast, label='Forecast', linestyle='--')
    ax.set_title('Accident Occurrence Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Accidents')
    ax.legend()
    st.pyplot(fig)

elif task == "Anomaly Detection":
    st.subheader("Anomaly Detection in Accident Severity Using IsolationForest")
    
    # Ensure that 'X_severity' is defined before using it here
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df_ml['Anomaly'] = iso_forest.fit_predict(X_severity)

    fig, ax = plt.subplots()
    sns.scatterplot(x='ACCIDENT_DATE', y='SEVERITY', hue='Anomaly', data=df_ml, palette='viridis', ax=ax)
    ax.set_title('Anomaly Detection in Accident Severity')
    st.pyplot(fig)

# Run the app with: streamlit run app.py

