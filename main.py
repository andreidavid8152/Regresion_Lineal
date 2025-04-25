# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.api as sm

# Cargar el dataset
dataset = pd.read_csv("data/loan_data.csv")

# Filtrar datos inválidos:
dataset = dataset[(dataset["person_emp_exp"] >= 0) & (dataset["person_income"] > 0)]

# Variables
X = dataset["person_emp_exp"].values.reshape(-1, 1)
y = dataset["person_income"].values

# Crear y entrenar el modelo
regressor = LinearRegression()
regressor.fit(X, y)

# Predecir
y_pred = regressor.predict(X)

# Visualizar resultados
plt.scatter(X, y, color="red", label="Datos reales")
plt.plot(X, y_pred, color="blue", label="Modelo lineal")
plt.title("Ingresos vs Experiencia")
plt.xlabel("Años de Experiencia")
plt.ylabel("Ingresos")
plt.legend()
plt.show()

# ----- ANALISIS -----

# Correlacion
r = np.corrcoef(X.flatten(), y)[0, 1]

# R-cuadrado y MAE
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
media_ingresos = np.mean(y)
mae_porcentaje = (mae / media_ingresos) * 100

# p-valor
X_const = sm.add_constant(X)
modelo_sm = sm.OLS(y, X_const).fit()
p_valor = modelo_sm.pvalues[1]

# Interpretacion
print(
    f"""

      MODELO DE REGRESION SIMPLE

      Para este modelo se han analizado dos variables del conjunto de datos de préstamos, con el objetivo de comprender cómo la experiencia laboral de una persona influye en sus ingresos. Se ha tomado como variable independiente los años de experiencia laboral (X) y como variable dependiente los ingresos (Y).

      El modelo obtenido tiene un coeficiente de correlación (r) de: {r:.4f}, lo que sugiere una relación positiva moderada entre la experiencia y el nivel de ingresos. Además, el coeficiente de determinación (r²) es de {r2:.4f}, indicando que aproximadamente el {r2 * 100:.1f}% de la variabilidad en los ingresos puede explicarse por la experiencia laboral.

      El error absoluto medio (MAE) calculado es de {mae:.2f}, lo cual representa en promedio una desviación del {mae_porcentaje:.2f}% respecto a los ingresos reales. Finalmente, el p-valor asociado al coeficiente de experiencia es de {p_valor:.4e}, lo que confirma que la variable experiencia tiene un impacto estadísticamente significativo sobre los ingresos.
"""
)
