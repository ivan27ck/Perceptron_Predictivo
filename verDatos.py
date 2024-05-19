import pandas as pd

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv('LosDatos.csv')

# Reemplazar las edades inválidas (-1) con NaN
df['edad'] = df['edad'].replace(-1, pd.NA)

# Resumen de la columna 'y'
y_counts = df['y'].value_counts()
print("Resumen de la columna 'y':")
print(y_counts)

# Análisis de la relación de otras columnas con 'y'
# Agrupar por 'y' y obtener estadísticas descriptivas para cada grupo
grouped = df.groupby('y').agg({
    'edad': ['count', 'mean', 'std'],
    'marital': 'value_counts',
    'education': 'value_counts',
    'balance': ['count', 'mean', 'std'],
    'housing': 'value_counts',
    'loan': 'value_counts'
})

print("\nAnálisis de la relación de otras columnas con 'y':")
print(grouped)
