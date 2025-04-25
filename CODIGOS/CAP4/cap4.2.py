"""
Este script calcula y visualiza el campo eléctrico de cargas puntuales en un plano 2D.

El campo eléctrico es una representación de la fuerza que una carga eléctrica ejerce sobre otras cargas en su vecindad. En este caso, se consideran dos cargas puntuales: una positiva y una negativa, ambas ubicadas en el origen del sistema de coordenadas.

El código realiza las siguientes tareas:

1. **Importación de bibliotecas**:
   - Se importan las bibliotecas `numpy` para cálculos numéricos y `matplotlib` para la visualización gráfica.

2. **Definición de la función `get_vfield_radial_2d`**:
   - Esta función toma como entrada un conjunto de puntos en el plano y la posición de una carga puntual.
   - Calcula el campo eléctrico en cada uno de esos puntos utilizando la ley de Coulomb, que describe cómo las cargas eléctricas interactúan entre sí.
   - Se asegura de evitar divisiones por cero al filtrar puntos que están demasiado cerca de la carga.

3. **Definición de la función `tidy_up_ax`**:
   - Esta función se encarga de ajustar los ejes del gráfico para que tengan la misma escala y se vean bien.

4. **Configuración de la cuadrícula**:
   - Se genera una cuadrícula de puntos en el plano 2D donde se calculará el campo eléctrico.

5. **Cálculo y visualización del campo eléctrico**:
   - Se calcula el campo eléctrico para una carga puntual positiva y se grafica.
   - Luego, se calcula el campo eléctrico para una carga puntual negativa (invirtiendo la dirección del campo) y se grafica.

6. **Guardado de gráficos**:
   - Los gráficos se guardan como archivos PDF para su posterior visualización.

Este script es útil para entender visualmente cómo se comporta el campo eléctrico alrededor de cargas puntuales y puede ser una herramienta educativa en el estudio de la electrostática.
"""

import numpy as np
import matplotlib.pyplot as plt

def get_vfield_radial_2d(p, p_charge):
    # p: puntos en los que se calcula el campo eléctrico.
    # p_charge: ubicación de una carga puntual.
    x, y = p[0] - p_charge[0], p[1] - p_charge[1]
    r = np.sqrt(x**2 + y**2)
    
    # Evitar dividir por un número muy pequeño.
    valid_idx = np.where(r > np.max(r) * 0.01)
    p, r = p[:, valid_idx].squeeze(), r[valid_idx].squeeze()
    
    vf = np.zeros(p.shape)
    vf[0] = (p[0] - p_charge[0]) / r**2
    vf[1] = (p[1] - p_charge[1]) / r**2
    vf = vf / (2 * np.pi)
    
    return vf, p

def tidy_up_ax(ax):
    ax.axis('equal')
    ax.axis('square')
    lim = 1.25
    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xticks((-1, 0, 1))
    ax.set_yticks((-1, 0, 1))

# Set up a grid to plot the vector field.
step = 0.1
x, y = np.meshgrid(np.arange(-1, 1 + step, step), np.arange(-1, 1 + step, step))

xs, ys = 0.0, 0.0
p = np.vstack((x.flatten(), y.flatten()))

scale = 6

# Positive point charge
vf_pos, p = get_vfield_radial_2d(p, (xs, ys))
fig = plt.figure(figsize=(4, 4))  # Aumentar el tamaño de la figura
plt.scatter(xs, ys, marker="+", color='black')
plt.quiver(p[0], p[1], vf_pos[0], vf_pos[1], angles='xy', scale_units='xy', scale=scale)

tidy_up_ax(plt.gca())
plt.title('Positive Point Charge')
plt.savefig('fig_ch4_single_charge_pos.pdf', bbox_inches='tight')
plt.show()

# Negative point charge
vf_neg = -vf_pos
fig = plt.figure(figsize=(4, 4))  # Aumentar el tamaño de la figura
plt.scatter(xs, ys, marker="_", color='black')
plt.quiver(p[0], p[1], vf_neg[0], vf_neg[1], angles='xy', scale_units='xy', scale=scale)

tidy_up_ax(plt.gca())
plt.title('Negative Point Charge')
plt.savefig('fig_ch4_single_charge_neg.pdf', bbox_inches='tight')
plt.show()