# ====================================================
# CÓDIGO: VISUALIZACIÓN DE FUERZA MAGNÉTICA SOBRE CARGA
# ----------------------------------------------------
# Este código calcula y grafica la fuerza magnética que 
# actúa sobre una partícula cargada en movimiento dentro
# de un campo magnético uniforme, ilustrando la relación:
#               F = q(v × B)
# Muestra:
# 1. Vector velocidad (gris)
# 2. Vector fuerza (negro)
# 3. Puntos de grid que representan campo B uniforme
# ====================================================

import numpy as np
import matplotlib.pyplot as plt

# ====================================================
# PARÁMETROS FÍSICOS
# ====================================================
q = 2          # Carga de la partícula [C]
x, y = -1, -1.5 # Posición inicial de la partícula [m]

# Componentes de velocidad [m/s]
vx, vy, vz = -1, 1, 0  # Movimiento en plano xy (vz=0)

# Componentes del campo magnético [T]
Bx, By, Bz = 0, 0, 1   # Campo solo en dirección z

# ====================================================
# CÁLCULO DE FUERZA MAGNÉTICA (F = q(v × B))
# ====================================================
Fx = q * (vy * Bz - vz * By)  # Componente x de la fuerza
Fy = q * (vz * Bx - vx * Bz)  # Componente y de la fuerza
# Fz = 0 (porque vz=0 y By=0)

# ====================================================
# CONFIGURACIÓN DE LA FIGURA
# ====================================================
fig = plt.figure(figsize=(3, 3))

# Graficar vector velocidad (gris)
plt.quiver(x, y, vx, vy, color='gray',
           angles='xy', scale_units='xy', scale=1)

# Graficar vector fuerza (negro)
plt.quiver(x, y, Fx, Fy, color='black',
           angles='xy', scale_units='xy', scale=1)

# ====================================================
# VISUALIZACIÓN DEL CAMPO MAGNÉTICO
# ====================================================
lim, step = 3, 1  # Límites y espaciado del grid

# Crear grid para representar campo B uniforme
xgrid, ygrid = np.meshgrid(
    np.arange(-lim, lim+step, step),
    np.arange(-lim, lim+step, step),
    indexing='ij'
)

# Graficar puntos del grid (representan campo B)
plt.scatter(xgrid, ygrid, marker='.', color='gray')

# Marcar posición de la partícula
plt.scatter(x, y, s=100, marker='o', color='black')

# ====================================================
# CONFIGURACIÓN FINAL DEL PLOT
# ====================================================
plt.legend(('velocity [m/s]', 'Force [N]', 'B field [T]'),
           framealpha=1)
plt.axis('square')  # Mantener proporción 1:1
plt.xlim((-lim-0.5, lim+0.5))
plt.ylim((-lim-0.5, lim+0.5))
plt.xticks((-lim, 0, lim))
plt.yticks((-lim, 0, lim))
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('fuerza_magnetica.png')  # Guardar imagen
plt.show()