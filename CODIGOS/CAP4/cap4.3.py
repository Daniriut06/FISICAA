"""
Este script calcula y visualiza el campo eléctrico de dos cargas puntuales en un plano 2D.

Se consideran dos cargas puntuales, que pueden ser positivas o negativas, y se visualiza el campo eléctrico resultante en el plano. También se analiza el caso de un dipolo, que consiste en una carga positiva y una carga negativa.

El código realiza las siguientes tareas:

1. **Importación de bibliotecas**: Se importan las bibliotecas `numpy` para cálculos numéricos y `matplotlib` para la visualización gráfica.

2. **Definición de la función `get_vfield_radial_2d`**: Esta función calcula el campo eléctrico en un conjunto de puntos dados la posición de una carga puntual.

3. **Definición de la función `tidy_up_ax`**: Ajusta los ejes del gráfico para que tengan la misma escala y se vean bien.

4. **Configuración de la cuadrícula**: Se genera una cuadrícula de puntos en el plano 2D donde se calculará el campo eléctrico.

5. **Cálculo y visualización del campo eléctrico**: Se calculan los campos eléctricos para dos cargas positivas, dos cargas negativas y un dipolo, y se grafican.

6. **Guardado de gráficos**: Los gráficos se guardan como archivos PDF para su posterior visualización.

Este script es útil para entender visualmente cómo se comporta el campo eléctrico alrededor de múltiples cargas puntuales y puede ser una herramienta educativa en el estudio de la electrostática.
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

def tidy_up_ax(ax):
    ax.axis('equal')  # Establece la misma escala en ambos ejes
    ax.axis('square')  # Hace que el gráfico sea cuadrado
    lim = 1.25  # Límite para los ejes
    ax.set_xlim((-lim, lim))  # Establece los límites del eje x
    ax.set_ylim((-lim, lim))  # Establece los límites del eje y
    ax.set_xlabel('x')  # Etiqueta del eje x
    ax.set_ylabel('y')  # Etiqueta del eje y
    ax.set_xticks((-1, 0, 1))  # Establece las marcas en el eje x
    ax.set_yticks((-1, 0, 1))  # Establece las marcas en el eje y

# Set up a grid to plot the vector field.
step = 0.1  # Paso para la cuadrícula
x, y = np.meshgrid(np.arange(-1, 1 + step, step), np.arange(-1, 1 + step, step))  # Crea una cuadrícula de puntos

# Posiciones de las cargas
xs0, ys0 = 0.55, 0.00  # Carga positiva 1
xs1, ys1 = -0.55, 0.00  # Carga positiva 2
p = np.vstack((x.flatten(), y.flatten()))  # Apila los puntos de la cuadrícula en una matriz

# Dos cargas positivas
vf0, p = get_vfield_radial_2d(p, (xs0, ys0))  # Calcula el campo eléctrico de la carga 1
vf1, p = get_vfield_radial_2d(p, (xs1, ys1))  # Calcula el campo eléctrico de la carga 2
vf = vf0 + vf1  # Suma los campos eléctricos de ambas cargas

fig = plt.figure(figsize=(4, 4))  # Crea una figura de tamaño 4x4
plt.scatter(xs0, ys0, marker="+", color='black')  # Dibuja la carga positiva 1
plt.scatter(xs1, ys1, marker="+", color='black')  # Dibuja la carga positiva 2
plt.quiver(p[0], p[1], vf[0], vf[1], angles='xy', scale_units='xy', scale=6)  # Dibuja el campo eléctrico

tidy_up_ax(plt.gca())  # Ajusta los ejes del gráfico
plt.title('Two Positive Charges')  # Título del gráfico
plt.savefig('fig_ch4_double_charge_pos.pdf', bbox_inches='tight')  # Guarda el gráfico como PDF
plt.show()  # Muestra el gráfico

# Dos cargas negativas
vf = (-vf0) + (-vf1)  # Calcula el campo eléctrico de dos cargas negativas
fig = plt.figure(figsize=(4, 4))  # Crea una figura de tamaño 4x4
plt.scatter(xs0, ys0, marker="_", color='black')  # Dibuja la carga negativa 1
plt.scatter(xs1, ys1, marker="_", color='black')  # Dibuja la carga negativa 2
plt.quiver(p[0], p[1], vf[0], vf[1], angles='xy', scale_units='xy', scale=6)  # Dibuja el campo eléctrico

tidy_up_ax(plt.gca())  # Ajusta los ejes del gráfico
plt.title('Two Negative Charges')  # Título del gráfico
plt.savefig('fig_ch4_double_charge_neg.pdf', bbox_inches='tight')  # Guarda el gráfico como PDF
plt.show()  # Muestra el gráfico

# Dipolo (una carga positiva y una carga negativa)
vf = vf0 + (-vf1)  # Calcula el campo eléctrico de un dipolo
fig = plt.figure(figsize=(4, 4))  # Crea una figura de tamaño 4x4
plt.scatter(xs0, ys0, marker="+", color='black')  # Dibuja la carga positiva
plt.scatter(xs1, ys1, marker="_", color='black')  # Dibuja la carga negativa
plt.quiver(p[0], p[1], vf[0], vf[1], angles='xy', scale_units='xy', scale=6)  # Dibuja el campo eléctrico

tidy_up_ax(plt.gca())  # Ajusta los ejes del gráfico
plt.title('Dipole')  # Título del gráfico
plt.savefig('fig_ch4_double_charge_dipole.pdf', bbox_inches='tight')  # Guarda el gráfico como PDF
plt.show()  # Muestra el gráfico