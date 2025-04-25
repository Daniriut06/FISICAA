"""
Este código visualiza elementos de corriente a lo largo de un anillo circular centrado en el origen.
Genera puntos de corriente en un círculo y los representa en un gráfico 3D con flechas que muestran
la dirección de la corriente. La corriente fluye en dirección tangencial al círculo.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def current_elements_along_circle(R=1.0, dphi=np.pi/20):
    """
    Genera elementos de corriente a lo largo de un anillo circular
    Parámetros:
        R: Radio del círculo
        dphi: Incremento angular en radianes
    Retorna:
        pos: Posiciones 3D de los elementos (3xN array)
        curr: Vectores de corriente (3xN array)
    """
    # Generar ángulos desde 0 hasta 2pi (no incluido)
    phi = np.arange(0, 2*np.pi, dphi)
    N = len(phi)
    
    # Inicializar arrays para posiciones y vectores de corriente
    pos = np.zeros((3, N))
    curr = np.zeros((3, N))
    
    # Calcular posiciones x, y (el círculo está en el plano z=0)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    
    # Asignar posiciones
    pos[0] = x  # coordenada x
    pos[1] = y  # coordenada y
    pos[2] = 0  # coordenada z (todas en z=0)
    
    # Los vectores de corriente son tangentes al círculo (-sin(phi), cos(phi), 0)
    curr[0] = -np.sin(phi)  # componente x
    curr[1] = np.cos(phi)   # componente y
    curr[2] = 0             # componente z
    
    # Normalizar los vectores de corriente
    curr = curr * dphi/(2*np.pi)
    
    return pos, curr

# Crear figura 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Generar elementos de corriente en un círculo de radio 1.0
R = 1.0
dphi = np.pi/20  # Paso angular de 9 grados
pos, curr = current_elements_along_circle(R, dphi)

# Dibujar el círculo completo (para referencia)
phi = np.linspace(0, 2*np.pi, 100)
x = R * np.cos(phi)
y = R * np.sin(phi)
ax.plot(x, y, np.zeros_like(x), 'k-', linewidth=2, alpha=0.5)

# Dibujar los elementos de corriente como flechas
ax.quiver(pos[0], pos[1], pos[2],
          curr[0], curr[1], curr[2],
          color='red', length=0.2, normalize=True,
          arrow_length_ratio=0.3)

# Configurar los ejes
ax.set_box_aspect([1, 1, 1])  # Aspecto igual en todas las direcciones
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Eje Z')
ax.set_title('Elementos de corriente en un anillo circular')

# Mostrar el gráfico
plt.tight_layout()
plt.show()