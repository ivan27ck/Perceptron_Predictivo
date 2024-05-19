import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('LosDatos.csv')
registro_prediccion = [None] * 6

def limpiar_pantalla():
    # Determina el sistema operativo y ejecuta el comando correspondiente
    if os.name == 'nt':  # Para Windows
        os.system('cls')
    else:  # Para macOS y Linux
        os.system('clear')



# Dividir el conjunto de datos en características de entrada (X) y la variable de salida (Y)
X = dataset.iloc[:, :6].values  
Y = dataset.iloc[:, -1].values 

# Normalizar los datos
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)
Y_normalized = scaler_Y.fit_transform(Y.reshape(-1, 1))


X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y_normalized, test_size=0.2, random_state=42)

# Definir el modelo con varias capas
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, input_shape=[6], activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1)
])


modelo.compile(optimizer='adam', loss='mean_squared_error')


print("Entrenamiento")
historial = modelo.fit(X_train, Y_train, epochs=10, verbose=False, validation_data=(X_test, Y_test))


import matplotlib.pyplot as plt
loss = historial.history['loss']

plt.plot(loss, label='Pérdida en entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Curva de pérdida durante el entrenamiento')
plt.legend()
plt.show()

print("Error del modelo")
loss_values = historial.history["loss"]
loss_values_float = [float(loss) for loss in loss_values]
for loss in loss_values_float:
    print(loss)

limpiar_pantalla()
print("Registro a predecir. Recuerda si = 1, no = 0.\n")
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

