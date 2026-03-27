
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide")
st.title("🌱 EcoVerge AI Dashboard")

menu = st.sidebar.radio("Navigation", [
    "EDA & Overview",
    "Segmentation",
    "Prediction",
    "Association Rules",
    "New Customer"
])

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    df_enc = df.copy()
    encoders = {}
    for col in df_enc.columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    if menu == "EDA & Overview":
        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Total Customers", len(df))
        col2.metric("Avg Spend", df["monthly_spend"].mode()[0])
        col3.metric("Switch %", round((df["switch_likelihood"]=="Very Likely").mean()*100,2))
        col4.metric("Top Segment", df["business_type"].mode()[0])

        tab1, tab2 = st.tabs(["Demographics","Behavior"])

        with tab1:
            st.plotly_chart(px.histogram(df, x="business_type"))
            st.plotly_chart(px.histogram(df, x="city_tier"))

        with tab2:
            st.plotly_chart(px.histogram(df, x="monthly_volume"))
            st.plotly_chart(px.histogram(df, x="monthly_spend"))

    elif menu == "Segmentation":
        X = df_enc.drop("switch_likelihood", axis=1)
        kmeans = KMeans(n_clusters=4, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X)
        st.plotly_chart(px.histogram(df, x="Cluster", color="Cluster"))

    elif menu == "Prediction":
        X = df_enc.drop("switch_likelihood", axis=1)
        y = df_enc["switch_likelihood"]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train,y_train)
        pred = model.predict(X_test)

        st.write("Accuracy:", accuracy_score(y_test,pred))
        st.write("Precision:", precision_score(y_test,pred,average="weighted"))
        st.write("Recall:", recall_score(y_test,pred,average="weighted"))
        st.write("F1:", f1_score(y_test,pred,average="weighted"))

    elif menu == "Association Rules":
        df_bool = df_enc.applymap(lambda x:1 if x>0 else 0)
        freq = apriori(df_bool, min_support=0.1, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=0.5)
        st.dataframe(rules[['antecedents','consequents','confidence','lift']])

    elif menu == "New Customer":
        X = df_enc.drop("switch_likelihood", axis=1)
        y = df_enc["switch_likelihood"]
        model = RandomForestClassifier()
        model.fit(X,y)

        inputs = {}
        for col in X.columns:
            inputs[col] = st.selectbox(col, encoders[col].classes_)

        if st.button("Predict"):
            input_df = pd.DataFrame([inputs])
            for col in input_df.columns:
                input_df[col] = encoders[col].transform(input_df[col])
            pred = model.predict(input_df)[0]
            st.success(f"Prediction: {pred}")
