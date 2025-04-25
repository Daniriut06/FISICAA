"""
Este script calcula y visualiza el campo eléctrico radial de una carga puntual en un plano 2D.

El campo eléctrico es una representación de la fuerza que una carga eléctrica ejerce sobre otras cargas en su vecindad. En este caso, se considera una carga puntual ubicada en el origen del sistema de coordenadas.

El código realiza las siguientes tareas:

1. **Importación de bibliotecas**:
   - Se importan las bibliotecas `numpy` para cálculos numéricos y `matplotlib` para la visualización gráfica.

2. **Definición de la función `get_vfield_radial_2d`**:
   - Esta función toma como entrada un conjunto de puntos en el plano y la posición de una carga puntual.
   - Calcula el campo eléctrico en cada uno de esos puntos utilizando la ley de Coulomb, que describe cómo las cargas eléctricas interactúan entre sí.
   - Se asegura de evitar divisiones por cero al filtrar puntos que están demasiado cerca de la carga.

3. **Creación de una cuadrícula de puntos**:
   - Se genera una cuadrícula de puntos en el plano 2D donde se calculará el campo eléctrico.

4. **Definición de la carga puntual**:
   - Se establece la posición de la carga puntual (en este caso, en el origen).

5. **Cálculo del campo eléctrico**:
   - Se llama a la función `get_vfield_radial_2d` para calcular el campo eléctrico en los puntos de la cuadrícula.

6. **Visualización**:
   - Se utiliza `matplotlib` para graficar el campo eléctrico como flechas (usando `quiver`), donde la dirección de las flechas indica la dirección del campo y la longitud indica su magnitud.
   - Se marca la posición de la carga puntual en el gráfico.

Este script es útil para entender visualmente cómo se comporta el campo eléctrico alrededor de una carga puntual y puede ser una herramienta educativa en el estudio de la electrostática.
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

# Crear una cuadrícula de puntos en 2D
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)
P = np.array([X.flatten(), Y.flatten()])

# Definir la posición de la carga puntual
p_charge = np.array([0, 0])  # Carga en el origen

# Calcular el campo eléctrico
vf, P_valid = get_vfield_radial_2d(P, p_charge)

# Graficar el campo eléctrico
plt.figure(figsize=(8, 8))
plt.quiver(P_valid[0], P_valid[1], vf[0], vf[1], color='r', headlength=5)
plt.scatter(p_charge[0], p_charge[1], color='blue', s=100, label='Carga puntual')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Campo Eléctrico Radial de una Carga Puntual')
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()
