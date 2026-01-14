import streamlit as st

import pandas as pd

from sklearn.datasets import load_iris

import numpy as np

from sklearn.ensemble import RandomForestClassifier

import time


st.title("Hi Ananta, Welcome...")

st.header("Header and Text")

st.subheader("This is Subheader..")

slider_val = st.slider("Select a value", 0, 100, 50)

st.write(f"Slider value: {slider_val}")

if st.button("Click Me"):
    st.write("Button Clicked!")
    
options = st.multiselect("Choose multiple", ["A", "B", "C"])

st.write("Selected:", options)

uploaded_file = st.file_uploader("Upload a File")



if uploaded_file is not None:
    st.write("File uploaded:", uploaded_file.name)
    
    df = pd.read_csv(uploaded_file)
    
    st.write("Uploaded Data")
    
    st.dataframe(df)
    
    #basic transformation
    
    if st.button("Show Summary"):
        st.write(df.describe())
        
    if st.button("Fileter by Column"):
        column = st.selectbox("Select column to filter", df.columns)
        
        value = st.text_input("Enter value to filter")
        
        if value:
            filtered = df[df[column].astype(str).str.contains(value,case = False)]
            st.dataframe(filtered)
            
    else:
        st.write("Please upload a CSV file.")
#radio 
    
radio_choice = st.radio("Chooaw one:", ["a", "b", "c", "d"])


st.write(f"Radio choice: {radio_choice}")

#checkbox

checkbox = st.checkbox("Check Me")

if checkbox:
    st.write("Checkbox is Checked")

#progress bar

progress_bar = st.progress(0)

status_text = st.empty()

for i in range(100):
    progress_bar.progress(i+1)
    
    status_text.text(f"Progress: {i+1}%")
    
    time.sleep(0.1)
    
status_text.text("Done")

#real-time chart

chart_placeholder = st.empty()

data = pd.DataFrame(np.random.randn(100, 1), columns=["A"])

chart_placeholder.line_chart(data)


iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["target"] = iris.target

st.write("Iris dataset")

st.dataframe(df.head())

#

model = RandomForestClassifier()

model.fit(iris.data, iris.target)

#

st.header("Make a prediction")

sepal_length = st.slider("Sepal length", 4.0, 8.0, 5.0)

sepal_width = st.slider("Sepal width", 4.0, 8.0, 5.0 )

petal_length = st.slider("Petal Length", 4.0, 8.0, 5.0 )

petal_width = st.slider("Petal width", 4.0, 8.0, 5.0  )



st.success("This message added after my first github Push.....")

st.caption("Updated on : 14 Jan 2026......")
