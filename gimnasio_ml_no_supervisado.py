# Somos un gimnasio, deseamos conocer los tipos de clientes
# En base a la edad y a la cantidad de horas a la semana
# Lo que se quiere es: Agrupar por tipo de cliente
# Tipo cliente:
# Cliente principiante
# Cliente regular
# Cliente avanzado
# Estamos practicando GIT - GITHUB - Para subir nuestro proyecto a la nube.
# Importar las librerías
import pandas as pd
from sklearn.cluster import KMeans

# Datos: 
# edad  - del cliente
# horas_entrenamiento  - Semanales
datos = {

   'edad': [

       18, 20, 19, 22,
       25, 27, 26, 28,
       35, 38, 40, 42

   ],

   'horas_entrenamiento': [

       1, 2, 1, 2,
       10, 12, 11, 15,
       28, 29, 30, 41

   ]
}

print(datos)
# Convertir estos datos a una tabla - matriz 
df = pd.DataFrame(datos)
print(df)

# Crear el modelo
modelo = KMeans(
    n_clusters=3,
    random_state=50
)

# Entrenar el modelo
modelo.fit(df)

# Etiquetas los grupos - Clusters
clusters = modelo.labels_
print('Grupos')
print(clusters)
df['Grupos'] = clusters
print('Nuevo Dataframe - Grupos')
print(df)

# Predicciones
nuevos_clientes = [
    [19,1],
    [28,5],
    [59,10]
]

predicciones = modelo.predict(nuevos_clientes)

print('Predicciones')
print(predicciones)
