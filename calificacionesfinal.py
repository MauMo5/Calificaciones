import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de calificaciones  ''')
st.image("imagencalificaciones.jpg", caption="Predicción de la calificación de un estudiante.")

st.header('Datos personales')

def user_input_features():
  # Entradas del usuario
  Horas_estudio = st.number_input('Horas de estudio:', min_value=0, max_value=100, value = 0 )
  Horas_sueño = st.number_input('Horas de sueño :',  min_value=0, max_value=100, value = 0 )
  Asistencia = st.number_input('Porcentaje de asistencia:', min_value=0, max_value=230, value = 0)
  Examen_previo = st.number_input('Calificación en el exámen previo:', min_value=0, max_value=140, value = 0)


#Utilizamos los nombres de nuestro conjunto de datos
  user_input_data = {'hours_studied': Horas_estudio,
                     'sleep_hours': Horas_sueño,
                     'attendance_percent': Asistencia,
                     'previous_scores': Examen_previo
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
datos =  pd.read_csv('exam_scores_df.csv', encoding='latin-1')
X = datos.drop(columns='exam_score')
y = datos['exam_score']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613797)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['hours_studied'] + b1[1]*df['sleep_hours'] + b1[2]*df['attendance_percent'] + b1[3]*df['previous_scores']

st.subheader('Predicción de la calificación')
st.write('La calificación del exámen es: ', prediccion)
