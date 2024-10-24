#Santos Nanny Eric Joan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Cargamos el archivo csv(olympics)
df = pd.read_csv('olympics.csv')

#Utilizamos unicamente la columna de medallas de oro ganadas
data = df['gold'].dropna()  # Elimina los valores NaN (si los hay)

# Calculo de la media
mean_value = np.mean(data)
print(f"Media: {mean_value}")

# Calculo de la mediana
median_value = np.median(data)
print(f"Mediana: {median_value}")

# Calculo de la moda
mode_value = stats.mode(data, keepdims=True).mode[0]
print(f"Moda: {mode_value}")

#Calculo de la varianza
variance_value = np.var(data)
print(f"Varianza: {variance_value}")

#Calculo de la función de distribución acumulada (CDF)
x = np.sort(data)
y = np.arange(1, len(x)+1) / len(x)

plt.figure(figsize=(10, 6))

# Graficar la CDF
plt.subplot(2, 1, 1)
plt.plot(x, y, marker='.', linestyle='none')
plt.title('Función de Distribución Acumulada (CDF)')
plt.xlabel('Valor')
plt.ylabel('Probabilidad Acumulada')

# Calculo de la función de dnsidad de probabilidad (PDF)
density = stats.gaussian_kde(data)
x_vals = np.linspace(min(data), max(data), 1000)

#Graficar la PDF
plt.subplot(2, 1, 2)
plt.plot(x_vals, density(x_vals))
plt.title('Función de Densidad de Probabilidad (PDF)')
plt.xlabel('Valor')
plt.ylabel('Densidad')

plt.tight_layout()
plt.show()
