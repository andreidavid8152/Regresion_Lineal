# Análisis Técnico y Estadístico del Modelo de Regresión Lineal

Este documento presenta una interpretación detallada de los resultados obtenidos en el modelo de regresión lineal simple entrenado para predecir ingresos en función de los años de experiencia laboral.

## 📈 Imagen 1 – Entrenamiento

### Interpretación:

En esta gráfica se observa la relación entre los años de experiencia laboral (`person_emp_exp`) y los ingresos (`person_income`) en el conjunto de entrenamiento.

- Los puntos rojos representan los datos reales del set de entrenamiento.
- La línea azul muestra la regresión lineal ajustada sobre estos datos.

### Análisis:
- Existe una **tendencia claramente positiva**: a mayor experiencia, mayores ingresos.
- La línea del modelo **se ajusta bastante bien a los datos**. Hay una dispersión razonable alrededor de la línea, pero sigue la tendencia general.
- Esto indica que el modelo aprendió correctamente la relación entre experiencia e ingresos en los datos de entrenamiento.

### Conclusión:
El modelo capta adecuadamente la tendencia lineal en los datos, lo que sugiere que la variable `experiencia` tiene una influencia directa y significativa en el ingreso dentro del conjunto de entrenamiento.


## 📊 Imagen 2 – Prueba

### Interpretación:

Esta gráfica muestra los resultados del modelo sobre el conjunto de prueba, es decir, datos que **no fueron utilizados** para entrenarlo.

- Los puntos rojos son los ingresos reales.
- La línea azul es la misma línea de regresión entrenada anteriormente.

### Análisis:
- Los puntos siguen la tendencia general de la línea azul, lo que indica que el modelo **generaliza bien** a nuevos datos.
- Hay cierta dispersión, pero **la mayoría de los puntos se alinean razonablemente** con el modelo.
- Esto sugiere que no hay un sobreajuste (overfitting), ya que el comportamiento se mantiene estable fuera del entrenamiento.

### Conclusión:
El modelo muestra **buen poder predictivo**, y su rendimiento sobre datos no vistos indica que la regresión lineal es una elección válida para esta relación.