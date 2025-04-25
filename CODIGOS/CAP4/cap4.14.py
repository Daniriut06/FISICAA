"""
Este código utiliza la biblioteca SymPy para realizar integraciones simbólicas de la función 
f(t) = (k - cos(t)) / (k² + 1 - 2k cos(t)) en el intervalo de 0 a 2π. 
Se evalúa la integral para dos casos: 
1. Cuando k = g + 1, asegurando que k > 1.
2. Cuando k = 1 - g, asegurando que k < 1.
El resultado de cada integración se muestra en la consola.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros generales
step = 0.001
x = np.arange(-1, 1, step)
fig = plt.figure(figsize=(12, 4))  # Más ancho para que se vea mejor
titles = ['(a) Circular', '(b) Shifted Oval', '(c) Arbitrary']

# Crear los tres contornos
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(titles[i])
    
    if i == 0:  # Circular
        y_top = np.sqrt(1 - x**2)
        y_bot = -y_top
    elif i == 1:  # Óvalo desplazado
        y_top = np.sqrt((1 - x**2) * 1.5) - 0.4
        y_bot = -np.sqrt((1 - x**2) * 1.5) - 0.4
    elif i == 2:  # Arbitrario
        y_top = 0.5 * x**3 + 0.5
        y_bot = x**2 + 0.5 * x - 0.5

    # Dibujar los contornos
    plt.plot(x, y_top, color='gray', linewidth=2)
    plt.plot(x, y_bot, color='gray', linewidth=2)
    plt.plot([1, 1], [y_top[-1], y_bot[-1]], color='gray', linewidth=2)

# Parámetros para rayos radiales
dtheta = np.pi / 16
ray = 5
theta = np.arange(0, 2 * np.pi, dtheta)

# Dibujar rayos radiales desde el origen
for i in range(3):
    plt.subplot(1, 3, i+1)
    for angle in theta:
        plt.plot([0, ray * np.cos(angle)], [0, ray * np.sin(angle)],
                 color='gray', linewidth=0.5)
    plt.scatter(0, 0, marker='+', color='black')  # Punto origen
    plt.axis('square')
    plt.axis('off')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])

plt.tight_layout()
plt.savefig('fig_ch4_gauss_law_boundary.pdf', bbox_inches='tight')
plt.show()
