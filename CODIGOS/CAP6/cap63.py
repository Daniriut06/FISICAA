# ====================================================
# CÓDIGO: SELECTOR DE VELOCIDAD (CAMPOS E y B CRUZADOS)
# ----------------------------------------------------
# Este código simula un selector de velocidad donde:
# - Campos eléctrico (E) y magnético (B) son perpendiculares
# - Solo partículas con velocidad v = E/B pasan sin desviarse
# Muestra:
# 1. Placas paralelas que generan campo E (líneas negras)
# 2. Puntos que representan campo B uniforme (gris)
# 3. Fuerzas eléctrica (negro) y magnética (gris) sobre carga
# 4. Trayectoria de partícula con velocidad v = E/B
# ====================================================

import numpy as np
import matplotlib.pyplot as plt

# Configuración de la figura
fig = plt.figure(figsize=(6,3))

# ====================================================
# CONFIGURACIÓN DEL ESPACIO Y CAMPOS
# ====================================================
step = 2
xlim, ylim = 4, 1.5  # Límites del área central

# Crear grid para representar campo B (puntos grises)
x_range = np.arange(-xlim, xlim+step, step)
xgrid, ygrid = np.meshgrid(
    x_range,
    np.arange(-ylim, ylim+0.5, 0.5),
    indexing='ij'
)
plt.scatter(xgrid, ygrid, marker='.', color='gray', label='B field')

# ====================================================
# PLACAS PARALELAS (GENERAN CAMPO ELÉCTRICO)
# ====================================================
# Placa superior (representada con guiones bajos '_')
xgrid_top, ygrid_top = np.meshgrid(x_range, +2.0, indexing='ij')
plt.scatter(xgrid_top, ygrid_top, marker='_', s=50, color='black')

# Placa inferior (representada con cruces '+')
xgrid_bot, ygrid_bot = np.meshgrid(x_range, -2.2, indexing='ij')
plt.scatter(xgrid_bot, ygrid_bot, marker='+', s=50, color='black')

# Bordes gruesos de las placas
plt.plot([-xlim, xlim], [+ylim, +ylim], color='black', linewidth=7)
plt.plot([-xlim, xlim], [-ylim, -ylim], color='black', linewidth=7)

# ====================================================
# PARTÍCULA Y FUERZAS
# ====================================================
x, y = -1, 0  # Posición inicial de la partícula
v = 3         # Velocidad que cumple v = E/B (en este ejemplo)
F = 1         # Magnitud de fuerza (arbitraria para visualización)

# Fuerza eléctrica (hacia arriba, negra)
plt.quiver(x, y, 0, +F, angles='xy', scale_units='xy', scale=1,
           linewidth=2, label='F (electric)', color='black')

# Fuerza magnética (hacia abajo, gris)
plt.quiver(x, y, 0, -F, angles='xy', scale_units='xy', scale=1,
           linewidth=2, label='F (magnetic)', color='gray')

# Vector velocidad (gris claro)
plt.quiver(x, y, +v, 0, angles='xy', scale_units='xy', scale=1,
           label='v = E/B', color='#CCCCCC')

# Partícula central
plt.scatter(x, y, s=100, marker='o', color='black')

# ====================================================
# PARTÍCULAS EN OTRAS POSICIONES (PARA CONTEXTO)
# ====================================================
# Partícula entrante por la izquierda
plt.quiver(-10, y, +v, 0, color='#CCCCCC',
           angles='xy', scale_units='xy', scale=1)
plt.scatter(-10, y, s=100, marker='o', color='black')

# Partícula saliente por la derecha
plt.quiver(9, y, +v, 0, color='#CCCCCC',
           angles='xy', scale_units='xy', scale=1)
plt.scatter(9, y, s=100, marker='o', color='black')

# ====================================================
# CONFIGURACIÓN FINAL DEL GRÁFICO
# ====================================================
plt.xlim((-13,15))
plt.ylim((-3,5))
plt.xticks(())  # Eliminar ticks en x
plt.yticks(())  # Eliminar ticks en y
plt.legend()
plt.tight_layout()
plt.savefig('selector_velocidad.pdf')
plt.show()