"""
Este script calcula y visualiza la aproximación del campo eléctrico de una línea infinita de cargas en un plano 2D.

Se utiliza una función para aproximar el campo eléctrico en diferentes rangos de integración en el eje y. Luego, se compara la magnitud del campo eléctrico resultante con el valor teórico.

El código realiza las siguientes tareas:

1. **Definición de la función `approx_line`**: Esta función calcula el campo eléctrico aproximado de una línea infinita de cargas.

2. **Configuración de parámetros**: Se definen los parámetros para la aproximación, como el paso en y y el rango máximo en y.

3. **Cálculo del campo eléctrico**: Se calcula el campo eléctrico para diferentes rangos máximos en y.

4. **Visualización**: Se grafican las magnitudes del campo eléctrico para diferentes rangos de integración.

5. **Comparación con el valor teórico**: Se imprime el valor numérico promedio y el valor teórico del campo eléctrico.
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

# Definición de la función para aproximar el campo eléctrico de una línea infinita
def approx_line(dy, y_max):
    dx = 0.1  # Paso en x
    x_range = np.arange(dx, 1 + dx, dx)  # Rango de valores en x
    p = np.vstack((x_range, np.zeros(len(x_range))))  # Crea una matriz de puntos en x
    vf = np.zeros(p.shape)  # Inicializa el campo eléctrico en cero
    y_range = np.arange(-y_max, y_max + dy, dy)  # Rango de valores en y

    # Cálculo del campo eléctrico
    for ys in y_range:
        vf0, _ = get_vfield_radial_2d(p, (0, ys))  # Calcula el campo eléctrico de cada carga en y
        vf += vf0 * dy  # Suma el campo eléctrico al total, multiplicado por dy

    return vf, p  # Devuelve el campo eléctrico y los puntos

# Parámetros para la aproximación
dy = 0.01  # Paso en y para la integración
y_max_range = (1, 2, 4, 8, 16, 1024)  # Diferentes rangos máximos en y

# Visualización
fig = plt.figure(figsize=(3, 3))  # Crea una figura de tamaño 3x3

# Cálculo y graficación del campo eléctrico para diferentes rangos
for y_max in y_max_range:
    vf, p = approx_line(dy, y_max)  # Calcula el campo eléctrico aproximado
    vf_mag = np.sqrt(np.sum(vf**2, axis=0))  # Calcula la magnitud del campo eléctrico
    plt.plot(p[0], vf_mag, color='gray')  # Grafica la magnitud del campo eléctrico
    plt.text(1.05, vf_mag[-1], "%d" % y_max)  # Añade texto con el rango máximo en y

# Ajustes del gráfico
plt.xlim((0, 1.3))  # Establece los límites del eje x
plt.ylim((0.2, 0.6))  # Establece los límites del eje y
plt.xticks((0, 0.5, 1))  # Establece las marcas en el eje x
plt.yticks((0.3, 0.4, 0.5))  # Establece las marcas en el eje y
plt.ylabel('|E|')  # Etiqueta para la magnitud del campo eléctrico
plt.xlabel('x')  # Etiqueta para el eje x

plt.savefig('fig_ch4_infinite_line_approx.pdf', bbox_inches='tight')  # Guarda el gráfico como PDF
plt.show()  # Muestra el gráfico

# Comparación con el valor teórico
print('Numerical approximation = %6.5f' % (np.mean(vf_mag)))  # Imprime el valor numérico promedio
rho = 1  # Densidad de carga
print('Theoretical value = %6.5f' % (rho / 2))  # Imprime el valor teórico del campo eléctrico