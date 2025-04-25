"""
Este script calcula y visualiza el campo eléctrico resultante de dos líneas paralelas de cargas en un plano 2D.

Se considera una línea infinita de cargas a lo largo de dos posiciones en el eje x, y se visualiza el campo eléctrico resultante en el plano. El campo eléctrico se calcula sumando el efecto de cada línea de carga.

El código realiza las siguientes tareas:

1. **Configuración de la cuadrícula**: Se genera una cuadrícula de puntos en el plano 2D donde se calculará el campo eléctrico.

2. **Definición de parámetros**: Se definen los parámetros para la posición de las líneas de carga y el rango de integración en el eje y.

3. **Cálculo del campo eléctrico**: Se calcula el campo eléctrico en cada punto de la cuadrícula debido a las dos líneas de carga.

4. **Visualización**: Se grafican las líneas de carga y el campo eléctrico resultante.

5. **Guardado del gráfico**: El gráfico se guarda como un archivo PDF para su posterior visualización.
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
x, y = np.meshgrid(np.arange(-1, 1 + step, step), np.arange(-1, 1 + step, step), indexing='ij')  # Crea una cuadrícula de puntos
p = np.vstack((x.flatten(), y.flatten()))  # Apila los puntos de la cuadrícula en una matriz
vf_double = np.zeros(p.shape)  # Inicializa el campo eléctrico en cero

# Parámetros para las líneas de carga
x0 = 0.55  # Posición de las líneas de carga
dy = 0.1  # Paso en y para la integración
y_max = 10  # Máximo valor en y para la línea infinita
y_range = np.arange(-y_max, y_max + dy, dy)  # Rango de valores en y

# Cálculo del campo eléctrico
for y0 in y_range:
    vf_r, _ = get_vfield_radial_2d(p, (+x0, y0))  # Campo eléctrico de la línea de carga derecha
    vf_l, _ = get_vfield_radial_2d(p, (-x0, y0))  # Campo eléctrico de la línea de carga izquierda
    vf_double += (vf_r - vf_l) * dy  # Suma el campo eléctrico resultante

# Visualización
fig = plt.figure(figsize=(4, 4))  # Crea una figura de tamaño 4x4
plt.scatter(np.zeros(y_range.shape) + x0, y_range, marker='+', color='black')  # Dibuja la línea de carga derecha
plt.scatter(np.zeros(y_range.shape) - x0, y_range, marker='_', color='black')  # Dibuja la línea de carga izquierda
plt.quiver(p[0], p[1], vf_double[0], vf_double[1], angles='xy', scale_units='xy', scale=16)  # Dibuja el campo eléctrico

# Ajustes del gráfico
plt.axis('equal')  # Establece la misma escala en ambos ejes
plt.axis('square')  # Hace que el gráfico sea cuadrado
lim = 1.0  # Límite para los ejes
plt.xlim((-lim, lim))  # Establece los límites del eje x
plt.ylim((-lim, lim))  # Establece los límites del eje y
plt.xlabel('x')  # Etiqueta para el eje x
plt.ylabel('y')  # Etiqueta para el eje y
plt.xticks((-1, 0, 1))  # Establece las marcas en el eje x
plt.yticks((-1, 0, 1))  # Establece las marcas en el eje y

plt.savefig('fig_ch4_two_infinite_lines.pdf', bbox_inches='tight')  # Guarda el gráfico como PDF
plt.show()  # Muestra el gráfico