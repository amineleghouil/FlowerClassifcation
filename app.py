import streamlit as stm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@stm.cache_data
def load_data():
    df = pd.read_csv("flower_dataset.csv")
    class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return df, class_names

df, class_names = load_data()

model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['classif'])

stm.sidebar.title("Input Features")
sl = stm.sidebar.slider("Sepal length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sw = stm.sidebar.slider("Sepal width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
pl = stm.sidebar.slider("Petal length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
pw = stm.sidebar.slider("Petal width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

in_data = [[sl, sw, pl, pw]]
pred = model.predict(in_data)

stm.write("### Prediction")
stm.write(f"The predicted species is: **{class_names[pred[0]]}**")
