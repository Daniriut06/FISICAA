"""
Este script calcula y visualiza el campo eléctrico resultante de una distribución de carga circular en un plano 2D.

Se utiliza una función para aproximar el campo eléctrico en un conjunto de puntos dados la posición de la carga circular. El campo eléctrico se calcula sumando el efecto de cada carga en la distribución circular.

El código realiza las siguientes tareas:

1. **Definición de la función `approx_circ`**: Esta función calcula el campo eléctrico aproximado de una distribución de carga circular.

2. **Configuración de parámetros**: Se definen los parámetros para la distribución circular y el rango de integración.

3. **Cálculo del campo eléctrico**: Se calcula el campo eléctrico en cada punto de la cuadrícula debido a la distribución circular de carga.

4. **Visualización**: Se grafican las flechas que representan el campo eléctrico resultante.

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

# Definición de la función para aproximar el campo eléctrico de una distribución circular
def approx_circ(p, r=1, d_phi=2 * np.pi / 200):
    phi_range = np.arange(0, 2 * np.pi, d_phi)  # Rango de ángulos
    vf = np.zeros(p.shape)  # Inicializa el campo eléctrico en cero
    for phi in phi_range:
        x0, y0 = r * np.cos(phi), r * np.sin(phi)  # Calcula la posición de la carga en la circunferencia
        vf0, _ = get_vfield_radial_2d(p, (x0, y0))  # Calcula el campo eléctrico de la carga
        vf += vf0 * r * d_phi  # Suma el campo eléctrico al total, multiplicado por el radio y el diferencial de ángulo
    return vf  # Devuelve el campo eléctrico total

# Parámetros para la visualización
d_phi = 2 * np.pi / 200  # Paso en el ángulo
scale = 1  # Escala de las flechas
lim = 4.5  # Límite para los ejes
step = 1  # Paso para la cuadrícula

# Configuración de la cuadrícula
x, y = np.meshgrid(np.arange(-lim, lim + step, step), np.arange(-lim, lim + step, step), indexing='ij')
p = np.vstack((x.flatten(), y.flatten()))  # Apila los puntos de la cuadrícula en una matriz

# Cálculo del campo eléctrico de la distribución circular
vf = approx_circ(p, r=1, d_phi=d_phi)

# Visualización
fig = plt.figure(figsize=(6, 6))  # Crea una figura de tamaño 6x6
plt.quiver(p[0], p[1], vf[0], vf[1], angles='xy', scale_units='xy', scale=scale)  # Dibuja el campo eléctrico

# Ajustes del gráfico
plt.xlim((-lim, lim))  # Establece los límites del eje x
plt.ylim((-lim, lim))  # Establece los límites del eje y
plt.xlabel('x')  # Etiqueta para el eje x
plt.ylabel('y')  # Etiqueta para el eje y
plt.title('Electric Field from a Circular Charge Distribution')  # Título del gráfico
plt.grid()  # Muestra la cuadrícula

plt.savefig('fig_ch4_circular_charge_distribution.pdf', bbox_inches='tight')  # Guarda el gráfico como PDF
plt.show()  # Muestra el gráfico