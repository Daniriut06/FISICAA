"""
Este script calcula y visualiza el campo eléctrico de una línea infinita de cargas en un plano 2D.

Se considera una línea infinita de cargas a lo largo del eje y, y se visualiza el campo eléctrico resultante en el plano. El campo eléctrico se calcula integrando el efecto de cada carga a lo largo de la línea.

El código realiza las siguientes tareas:

1. **Configuración de la cuadrícula**: Se genera una cuadrícula de puntos en el plano 2D donde se calculará el campo eléctrico.

2. **Filtrado de puntos**: Se filtran los puntos que están cerca del eje y para evitar singularidades en el cálculo del campo eléctrico.

3. **Definición del rango de integración**: Se define un rango de valores en el eje y para simular la línea infinita de cargas.

4. **Cálculo del campo eléctrico**: Se calcula el campo eléctrico en cada punto de la cuadrícula debido a la línea infinita de cargas.

5. **Visualización**: Se grafican las cargas y el campo eléctrico resultante.

6. **Guardado del gráfico**: El gráfico se guarda como un archivo PDF para su posterior visualización.
"""

import numpy as np  # Importa la biblioteca NumPy para cálculos numéricos
import matplotlib.pyplot as plt  # Importa Matplotlib para visualización gráfica

def get_vfield_radial_2d(p, p_charge):
    # p: puntos en los que se calcula el campo eléctrico.
    # p_charge: ubicación de una carga puntual.
    x, y = p[0] - p_charge[0], p[1] - p_charge[1]  # Calcula la distancia desde la carga
    r = np.sqrt(x**2 + y**2)  # Calcula la distancia radial

    # Evitar dividir por un número muy pequeño.
    valid_idx = np.where(r > np.max(r) * 0.01)  # Filtra puntos muy cercanos a la carga
    p, r = p[:, valid_idx].squeeze(), r[valid_idx].squeeze()  # Filtra los puntos válidos

    vf = np.zeros(p.shape)  # Inicializa el campo eléctrico
    vf[0] = (p[0] - p_charge[0]) / r**2  # Componente x del campo eléctrico
    vf[1] = (p[1] - p_charge[1]) / r**2  # Componente y del campo eléctrico
    vf = vf / (2 * np.pi)  # Normaliza el campo eléctrico

    return vf, p  # Devuelve el campo eléctrico y los puntos válidos

# Configuración de la cuadrícula
step = 0.1  # Paso para la cuadrícula
x, y = np.meshgrid(np.arange(-1, 1 + step, step), np.arange(-1, 1 + step, step))  # Crea una cuadrícula de puntos
p = np.vstack((x.flatten(), y.flatten()))  # Apila los puntos de la cuadrícula en una matriz

# Filtrado de puntos cerca del eje y
idx = np.where(np.abs(p[0]) > 0.001)  # Encuentra índices donde x es diferente de 0
p = np.vstack((p[0, idx], p[1, idx]))  # Filtra los puntos válidos

# Definición del rango de integración
dy = 0.1  # Paso en y para la integración
y_max = 10  # Máximo valor en y para la línea infinita
y_range = np.arange(-y_max, y_max, dy)  # Rango de valores en y

# Inicializa el campo eléctrico
vf_single = np.zeros(p.shape)  # Inicializa el campo eléctrico en cero

# Cálculo del campo eléctrico
for ys in y_range:
    vf0, p = get_vfield_radial_2d(p, (0, ys))  # Calcula el campo eléctrico de cada carga en y
    vf_single += vf0 * dy  # Suma el campo eléctrico al total, multiplicado por dy

# Visualización
fig = plt.figure(figsize=(4, 4))  # Crea una figura de tamaño 4x4
plt.scatter(np.zeros(y_range.shape), y_range, marker='+', color='black')  # Dibuja la línea de cargas
plt.quiver(p[0], p[1], vf_single[0], vf_single[1], angles='xy', scale_units='xy')  # Dibuja el campo eléctrico

# Ajustes del gráfico
plt.axis('equal')  # Establece la misma escala en ambos ejes
plt.axis('square')  # Hace que el gráfico sea cuadrado
plt.xlabel('x')  # Etiqueta del eje x
plt.ylabel('y')  # Etiqueta del eje y
lim = 1.0  # Límite para los ejes
plt.xlim((-lim, lim))  # Establece los límites del eje x
plt.ylim((-lim, lim))  # Establece los límites del eje y
plt.xticks((-1, 0, 1))  # Establece las marcas en el eje x
plt.yticks((-1, 0, 1))  # Establece las marcas en el eje y

plt.savefig('fig_ch4_infinite_line.pdf', bbox_inches='tight')  # Guarda el gráfico como PDF
plt.show()  # Muestra el gráfico