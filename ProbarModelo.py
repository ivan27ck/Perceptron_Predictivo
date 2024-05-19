import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


dataset = pd.read_csv('LosDatos.csv')
registro_prediccion = [None] * 6

# Dividir el conjunto de datos en características de entrada (X) y la variable de salida (Y)
X = dataset.iloc[:, :6].values  
Y = dataset.iloc[:, -1].values 

# Normalizar los datos
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)
Y_normalized = scaler_Y.fit_transform(Y.reshape(-1, 1))

modelo = tf.keras.models.load_model('modelo_entrenado.h5')

print("Registro a predecir. Recuerda: si = 1, no = 0.\n")
registro_prediccion[0]=input("Edad\n")
registro_prediccion[1]=input("Marital. soltero 0, casado 1, divorciado 2\n")
registro_prediccion[2]=input("Education. primary 1, secundary 2, superior 3, desconocido 4\n")
registro_prediccion[3]=input("Balance\n")
registro_prediccion[4]=input("Hipoteca\n")
registro_prediccion[5]=input("¿Ya tiene un prestamo?\n")


# Predecir valores
X_new = np.array([registro_prediccion])  
X_new_normalized = scaler_X.transform(X_new)
resultado_normalized = modelo.predict(X_new_normalized)
resultado = scaler_Y.inverse_transform(resultado_normalized)
resultado_redondeado = round(resultado[0][0] * 100, 1)
print("La probabilidad de que este cliente acepte un prestamo es:", resultado_redondeado, "%")

