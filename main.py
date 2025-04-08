# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Cargar el dataset
dataset = pd.read_csv("data/loan_data.csv")

# Filtrar datos inválidos:
# - experiencia laboral debe ser >= 0
# - ingresos deben ser > 0
dataset = dataset[(dataset["person_emp_exp"] >= 0) & (dataset["person_income"] > 0)]

# Variable independiente: experiencia laboral
X = dataset["person_emp_exp"].values.reshape(-1, 1)

# Variable dependiente: ingresos
y = dataset["person_income"].values

# Dividir en entrenamiento (2/3) y prueba (1/3)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=0
)

# Crear y entrenar modelo de regresión lineal
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predecir ingresos en el conjunto de prueba
y_pred = regressor.predict(X_test)

# Visualizar resultados - Entrenamiento
plt.scatter(X_train, y_train, color="red", label="Datos reales")
plt.plot(X_train, regressor.predict(X_train), color="blue", label="Modelo lineal")
plt.title("Ingresos vs Experiencia (Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Ingresos")
plt.legend()
plt.show()

# Visualizar resultados - Prueba
plt.scatter(X_test, y_test, color="red", label="Datos reales")
plt.plot(X_train, regressor.predict(X_train), color="blue", label="Modelo lineal")
plt.title("Ingresos vs Experiencia (Prueba)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Ingresos")
plt.legend()
plt.show()
