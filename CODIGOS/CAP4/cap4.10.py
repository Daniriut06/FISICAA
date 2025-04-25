"""
Este código genera un diagrama que representa un círculo con un radio R y una distancia D en un sistema de coordenadas cartesianas. 
El diagrama incluye líneas que conectan el origen O con un punto P en el círculo, así como etiquetas que indican las distancias y ángulos relevantes. 
El gráfico se guarda como un archivo PDF y se muestra en pantalla.
"""

import numpy as np  # Importa la biblioteca NumPy para cálculos numéricos
import matplotlib.pyplot as plt  # Importa Matplotlib para visualización gráfica

# Parámetros del diagrama
D = 3  # Distancia en el eje x desde el origen O hasta el punto Q
R = 1  # Radio del círculo
step = 0.01  # Paso para el rango de ángulos, determina la resolución del círculo
phi_range = np.arange(0, 2 * np.pi + step, step)  # Rango de ángulos de 0 a 2π para dibujar el círculo

# Ángulo específico
phi = np.pi / 4  # Ángulo en radianes para calcular las coordenadas del punto P

# Crear figura
fig = plt.figure(figsize=(5, 3))  # Inicializa una figura con un tamaño de 5x3 pulgadas

# Dibujar el círculo
plt.plot(R * np.cos(phi_range), R * np.sin(phi_range), color='gray', linewidth=4)  # Dibuja el círculo en gris

# Calcular coordenadas del punto P
x, y = R * np.cos(phi), R * np.sin(phi)  # Calcula las coordenadas cartesianas del punto P en el círculo

# Dibujar líneas
plt.plot((0, x), (0, y), color='black')  # Dibuja la línea OP desde el origen O hasta el punto P
plt.plot((x, D), (y, 0), color='black')  # Dibuja la línea PQ desde el punto P hasta el punto Q en el eje x
plt.plot((0, D), (0, 0), color='black')  # Dibuja la línea OQ desde el origen O hasta el punto Q
plt.plot((x, x), (0, y), color='gray')  # Dibuja una línea vertical en el punto P

# Añadir etiquetas
plt.text(x - 0.5, y - 0.3, 'R')  # Etiqueta para el radio R
plt.text(D, -0.15, 'D')  # Etiqueta para la distancia D
plt.text(-0.1, -0.15, 'O')  # Etiqueta para el origen O
plt.text(x, y + 0.1, 'P')  # Etiqueta para el punto P
plt.text(x, -0.15, 'Q')  # Etiqueta para el punto Q
plt.text(D - 1.5, +0.6, 'r')  # Etiqueta para la distancia r
plt.text(0.2, 0.07, r'$\phi$')  # Etiqueta para el ángulo φ
plt.text(D - 0.7, 0.07, r'$\theta$')  # Etiqueta para el ángulo θ

# Ajustes de la visualización
plt.axis('square')  # Mantiene la proporción del gráfico para que el círculo se vea como un círculo
plt.axis('off')  # Desactiva los ejes para una visualización más limpia
plt.xlim(np.array([-R, D]) * 1.1)  # Limita el eje x para que se ajuste al gráfico
plt.ylim(np.array([-1, 1]) * R * 1.1)  # Limita el eje y para que se ajuste al gráfico

# Guardar y mostrar la figura
plt.savefig('fig_ch4_circle_diagram.pdf')  # Guarda el gráfico como un archivo PDF
plt.show()  # Muestra el gráfico en pantalla