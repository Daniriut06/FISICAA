# ====================================================
# CÓDIGO: MOVIMIENTO CIRCULAR DE CARGA EN CAMPO MAGNÉTICO
# ----------------------------------------------------
# Este código visualiza cómo una partícula cargada se mueve
# en trayectoria circular bajo un campo magnético uniforme,
# ilustrando la fuerza de Lorentz (F = qv×B) que actúa como
# fuerza centrípeta. Muestra:
# 1. Múltiples posiciones en la órbita circular
# 2. Vectores velocidad (gris) y fuerza (negro) en cada punto
# 3. Grid que representa el campo magnético uniforme B
# ====================================================

import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial de la figura
fig = plt.figure(figsize=(5,5))

# ====================================================
# PARÁMETROS FÍSICOS
# ====================================================
phi_range = np.arange(0, 2*np.pi, 2*np.pi/7)  # Ángulos para 7 posiciones
v = -1            # Magnitud de la velocidad [m/s]
q = 1             # Carga de la partícula [C]
R = 1.75          # Radio de la órbita circular [m]
# Relación física: m = qBR/v (masa implícita)

# Configuración del grid para representar campo B
lim, step = 3, 1  # Límites y espaciado del grid
xgrid, ygrid = np.meshgrid(
    np.arange(-lim, lim+step, step),
    np.arange(-lim, lim+step, step),
    indexing='ij'
)

# ====================================================
# CÁLCULO PARA CADA POSICIÓN EN LA ÓRBITA
# ====================================================
for phi in phi_range:
    # Posición circular (coordenadas polares → cartesianas)
    x, y = R*np.cos(phi), R*np.sin(phi)
    
    # Vector velocidad (tangente a la circunferencia)
    vx, vy, vz = -np.sin(phi)*v, np.cos(phi)*v, 0
    
    # Campo magnético uniforme (solo componente z)
    Bx, By, Bz = 0, 0, 1
    
    # Fuerza de Lorentz (F = q(v × B))
    Fx = q*(vy*Bz - vz*By)
    Fy = q*(vz*Bx - vx*Bz)
    
    # ====================================================
    # VISUALIZACIÓN
    # ====================================================
    # Graficar vector velocidad (gris)
    plt.quiver(x, y, vx, vy, color='gray',
               angles='xy', scale_units='xy', scale=1)
    
    # Graficar vector fuerza (negro)
    plt.quiver(x, y, Fx, Fy, color='black',
               angles='xy', scale_units='xy', scale=1)
    
    # Graficar grid de campo B y posición de la carga
    plt.scatter(xgrid, ygrid, marker='.', color='gray')
    plt.scatter(x, y, s=100, marker='o', color='black')

# ====================================================
# CONFIGURACIÓN FINAL DEL GRÁFICO
# ====================================================
plt.legend(('velocity [m/s]', 'Force [N]', 'B field [T]'),
           framealpha=1)
plt.axis('square')  # Mantener relación de aspecto 1:1
plt.xlim((-lim-0.5, lim+0.5))
plt.ylim((-lim-0.5, lim+0.5))
plt.xticks((-lim, 0, lim))
plt.yticks((-lim, 0, lim))
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.tight_layout()
plt.savefig('orbita_circular_campo_magnetico.png')
plt.show()