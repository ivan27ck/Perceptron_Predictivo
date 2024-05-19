import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
# Asegúrate de cambiar 'tu_archivo.csv' por la ruta real de tu archivo CSV
df = pd.read_csv('LosDatos.csv', header=None)

# Asignar nombres a las columnas si es necesario
df.columns = ['Edad', 'Marital', 'Educacion', 'Saldo', 'Hipoteca', 'Prestamo', 'target']

# Mostrar las primeras filas del DataFrame para verificar la carga de datos
print(df.head())

# Crear una nueva columna categorizando la cuarta columna
df['saldo_nuevo'] = df['Saldo'].apply(lambda x: 'Mayor a 0' if x > 0 else 'Menor o igual a 0')

# Función para graficar la distribución de la columna objetivo
def plot_distribution(column_name):
    distribution = df.groupby([column_name, 'target']).size().unstack(fill_value=0)
    distribution = distribution.div(distribution.sum(axis=1), axis=0)
    distribution.plot(kind='bar', stacked=True)
    plt.title(f'Distribución de la columna objetivo para {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Proporción')
    plt.legend(title='Target')
    plt.show()

#mostrar los resultados
for col in df.columns[:-2]:  # Excluir la columna objetivo
    print(f"\nDistribución de valores en 'target' para cada valor en '{col}':")
    distribution = df.groupby(col)['target'].value_counts(normalize=True).unstack()
    print(distribution)

# Graficar para cada columna excepto la categorizada, la columna objetivo y la columna Saldo
for col in df.columns[:-5]:  # Excluir la columna Saldo y la columna objetivo
    plot_distribution(col)

# Graficar para la nueva columna categorizada
plot_distribution('saldo_nuevo')

# Graficar para las columnas Hipoteca y Prestamo
for col in df.columns[4:-1]:  # Excluir la columna objetivo
    plot_distribution(col)
