import streamlit as stm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

@stm.cache_data
def load_data():
    df = pd.read_csv("flower_dataset.csv")
    class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return df, class_names

df, class_names = load_data()

X = df.iloc[:, :-1]
y = df['classif']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

stm.sidebar.title("Input Features")
sl = stm.sidebar.slider("Sepal length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sw = stm.sidebar.slider("Sepal width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
pl = stm.sidebar.slider("Petal length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
pw = stm.sidebar.slider("Petal width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

in_data = [[sl, sw, pl, pw]]
pred = model.predict(in_data)

stm.write("### 🌸 Prediction")
stm.write(f"The predicted species is: **{class_names[pred[0]]}**")

stm.write("### 📊 Model Performance")
stm.write(f"Model Accuracy on Test Set: **{accuracy * 100:.2f}%**")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names.values(), yticklabels=class_names.values(), ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
stm.pyplot(fig)
