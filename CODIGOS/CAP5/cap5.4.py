"""
Este código crea y visualiza puntos en un sistema de coordenadas cartesianas 3D.
Genera una malla de puntos espaciados uniformemente dentro de un cubo centrado en el origen,
y luego muestra estos puntos en un gráfico 3D con etiquetas y ejes apropiados.
"""

import numpy as np
import matplotlib.pyplot as plt

def puntos_en_cartesianas(L=1.0, delta=0.1):
    """
    Genera puntos en coordenadas cartesianas 3D dentro de un cubo de lado 2L
    
    Parámetros:
        L: Mitad de la longitud del lado del cubo (el cubo va de -L a L)
        delta: Espaciado entre puntos
        
    Retorna:
        Matriz 3xN con las coordenadas (x,y,z) de los puntos
    """
    # Crear vector desde -L hasta L con paso delta
    v = np.arange(-L, L + delta, delta)
    
    # Crear malla 3D de puntos
    x, y, z = np.meshgrid(v, v, v, indexing='ij')
    
    # Apilar y aplanar las coordenadas para obtener matriz 3xN
    P = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    
    return P

def plot_coordenadas(P, titulo=""):
    """
    Visualiza puntos en un sistema de coordenadas 3D
    
    Parámetros:
        P: Matriz 3xN con coordenadas (x,y,z) de los puntos
        titulo: Título del gráfico
    """
    # Crear figura 3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar puntos
    ax.scatter(P[0], P[1], P[2], color='black')
    
    # Configurar etiquetas de ejes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # Configurar límites de ejes
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    # Configurar título
    ax.set_title(titulo)
    
    # Ajustar diseño y mostrar
    plt.tight_layout()

# Generar puntos en coordenadas cartesianas
L = 1.0  # Mitad del tamaño del cubo
delta = 0.2  # Espaciado entre puntos
P = puntos_en_cartesianas(L, delta)

# Visualizar los puntos
plot_coordenadas(P, titulo="Puntos en Coordenadas Cartesianas")

# Mostrar el gráfico
plt.show()