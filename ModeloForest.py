import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Título principal
st.title("Predicción de Riesgo Financiero")

# Cargar los datos en caché
@st.cache_data
def cargar_datos():
    ds = pd.read_csv("dataset_financiero_riesgo.csv")
    return ds

ds = cargar_datos()
st.write("Vista previa de los datos")
st.dataframe(ds.head())

# Copia del dataset original para codificación
ds_encode = ds.copy()

# Codificadores separados para cada variable categórica
le_historial = LabelEncoder()
le_educacion = LabelEncoder()

ds_encode["Historial_Credito"] = le_historial.fit_transform(ds["Historial_Credito"])
ds_encode["Nivel_Educacion"] = le_educacion.fit_transform(ds["Nivel_Educacion"])

# Variables predictoras y variable objetivo
x = ds_encode.drop("Riesgo_Financiero", axis=1)
y = LabelEncoder().fit_transform(ds_encode["Riesgo_Financiero"])

# División de los datos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Entrenamiento del modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=0)
modelo.fit(x_train, y_train)
score = modelo.score(x_test, y_test)

# Precisión del modelo
st.subheader(f"Precisión del modelo: {score:.2f}")

# Matriz de confusión
y_pred = modelo.predict(x_test)
mc = confusion_matrix(y_test, y_pred)

st.subheader('Matriz de Confusión')
fig, ax = plt.subplots()
sns.heatmap(mc, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Importancia de características
importancias = modelo.feature_importances_
st.subheader("Importancia de las características")
importancia_ds = pd.DataFrame({"Característica": x.columns, "Importancia": importancias})
st.bar_chart(importancia_ds.set_index("Característica"))

# Formulario de predicción
st.subheader("Formulario de Predicción")
with st.form("formulario"):
    ingresos = st.number_input("Ingresos mensuales", min_value=0.0, max_value=3000.0)
    gastos = st.number_input("Gastos mensuales", min_value=0.0, max_value=2000.0)
    deudas = st.slider("Deudas Activas", 0, 5, 2)
    historial = st.selectbox("Historial de Crédito", ["Bueno", "Regular", "Malo"])
    edad = st.slider("Edad", 21, 64, 30)
    tarjeta = st.radio("¿Tiene tarjeta de crédito?", [0, 1])
    educacion = st.selectbox("Nivel de Educación", ["Basico", "Medio", "Superior"])
    inversiones = st.slider("Inversiones Activas", 0, 3, 1)

    submit = st.form_submit_button("Predecir")

    if submit:
        historial_cod = le_historial.transform([historial])[0]
        educacion_cod = le_educacion.transform([educacion])[0]

        entrada = pd.DataFrame([[ingresos, gastos, deudas, historial_cod, edad, tarjeta, educacion_cod, inversiones]],
                               columns=x.columns)

        pred = modelo.predict(entrada)[0]
        riesgo = {0: "Alto", 1: "Bajo", 2: "Medio"}.get(pred, "Desconocido")
        st.success(f"Nivel de Riesgo Financiero según la predicción: {riesgo}")
