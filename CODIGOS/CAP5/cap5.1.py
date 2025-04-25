import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def currents_along_line(I, L, dL, x0=0, y0=0):
    # Distribución de corriente a lo largo de un alambre recto vertical.
    vec = np.arange(-L/2, L/2 + dL, dL)
    N = len(vec)
    p = np.zeros((3, N))
    p[0], p[1] = x0, y0
    p[2] = vec
    curr = np.zeros((3, N))
    curr[2] = I * dL
    return p, curr

def tidy_ax(ax):
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax

# Parámetros de corriente
I = 1  # Corriente total (amperios, por ejemplo)
L = 1  # Longitud del alambre
dL = 0.05  # Tamaño del segmento

# Obtener puntos y vectores de corriente
p, curr = currents_along_line(I, L, dL)

# Visualización 3D
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(p[0], p[1], p[2], curr[0], curr[1], curr[2], length=0.05, normalize=True, color='r')
tidy_ax(ax)
plt.title("Distribución de corriente en un alambre recto")
plt.show()
