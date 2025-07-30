import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Mostrar o cargar los datos
ds = pd.read_csv("dataset_financiero_riesgo.csv")

# Colocar un titulo principal en la página Web
st.title("Predicción de Riesgo Financiero")

# Cargar los datos en la memoria CACHE para mejorar la velocidad del acceso al 
# conjunto de datos
@st.cache_data 

# Hacemos una funcion que se llama cargar_datos. Leemos el archivo en una variable
# y retornamos la variable al que llama a la función. En este caso la variable 
# que retornamos se llama "ds", abreviatura de "dataset".
def cargar_datos():
    ds = pd.read_csv("dataset_financiero_riesgo.csv")
    return ds

ds = cargar_datos()
st.write("Vista previa de los datos")
st.dataframe(ds.head())

# Preprocesamiento de datos o del conjunto de datos
ds_encode = ds.copy() # Copia el dataset completo a otro dataset

label_cols = ['Historial_Credito', 'Nivel_Educacion']
le = LabelEncoder()
for col in label_cols:
    ds_encode[col] = le.fit_transform(ds_encode[col])
    
x = ds_encode.drop("Riesgo_Financiero", axis=1)
y = ds_encode["Riesgo_Financiero"]
y = LabelEncoder().fit_transform(y)

# Divifir el conjunto de datos en entrenamiento y testing
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# Entrenar el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=0)
modelo.fit(x_train, y_train)
score = modelo.score(x_test, y_test)

st.subheader(f"Precision del modelo: {score:.2f}")

# Matriz de confusion
y_pred = modelo.predict(x_test)
mc = confusion_matrix(y_test, y_pred)
st.subheader('Matriz de Confusión')
fig, ax = plt.subplots()

sns.heatmap(mc, annot=True, fmt='d', cmap='Blues', ax=ax )
st.pyplot(fig)

# Importancia de las características
importancias = modelo.feature_importances_
st.subheader("Importancia de las características")
importancia_ds = pd.DataFrame({"Característica": x.columns, "Importancia":importancias})
st.bar_chart(importancia_ds.set_index("Característica"))

# Formulario de predicción
st.subheader("Formulario de Predicción")
with st.form("formulario"):
    ingresos = st.number_input("Ingresos mensuales", min_value=0.0, max_value=3000.0)
    gastos = st.number_input("Gastos mensuales", min_value=0.0, max_value=2000.0)
    deudas = st.slider("Deudas Activas", 0,5,2)
    historial = st.selectbox("Historial Credito", ["Bueno", "Regular", "Malo"])
    edad = st.slider("Edad",21,64,30)
    tarjeta = st.radio("Tiene tarjeta de credito?", [0,1])
    educacion = st.selectbox("Nivel de Educacion", ["Basico","Medio","Superior"])
    inversiones = st.slider("Inversiones Activas", 0,3,1)
    
    #Crear un boton que nos diga: "PREDECIR"
    submit = st.form_submit_button("Predecir")
    
    if submit:
        historial_cod = le.fit(ds["Historial_Credito"]).fit_transform(historial)[0] 
        educacion_cod = le.fit(ds["Nivel_Educacion"]).fit_transform(educacion)[0]        
        entrada = pd.DataFrame([[ingresos, gastos, deudas, historial_cod, edad, tarjeta, educacion_cod, inversiones]], columns=x.columns)
        
        pred = modelo.predict(entrada)[0]
        riesgo={0:"Alto", 1:"Bajo", 2:"Medio"}.get(pred,"Desconocido")
        st.success(f"Nivel de Riesgo Financiero de acuerdo a la prediccion:{riesgo}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






