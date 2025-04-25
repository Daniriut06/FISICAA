# ====================================================
# CÓDIGO: EXPERIMENTO DE THOMSON (DEFLEXIÓN DE HAZ DE ELECTRONES)
# ----------------------------------------------------
# Simula la deflexión de un haz de electrones en un campo magnético
# cuando el campo eléctrico está desactivado, mostrando:
# 1. Placas paralelas sin carga (campo E apagado)
# 2. Campo magnético uniforme (puntos grises)
# 3. Trayectoria curva del electrón debido a la fuerza magnética
# ====================================================

import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial
fig = plt.figure(figsize=(6, 3))
step = 2
xlim, ylim = 4, 1.5  # Dimensiones del área de placas

# ====================================================
# VISUALIZACIÓN DEL CAMPO MAGNÉTICO (B)
# ====================================================
xgrid, ygrid = np.meshgrid(
    np.arange(-xlim, xlim+step, step),
    np.arange(-ylim, ylim+0.5, 0.5),
    indexing='ij'
)
plt.scatter(xgrid, ygrid, marker='.', color='gray', label='B field')

# ====================================================
# PLACAS PARALELAS (CAMPO ELÉCTRICO APAGADO)
# ====================================================
# Dibuja placas conductoras (sin carga)
plt.plot([-xlim, xlim], [-ylim, -ylim], 
         color='black', linewidth=7)
plt.plot([-xlim, xlim], [ylim, ylim], 
         color='black', linewidth=7)
plt.text(0, ylim+0.5, 'L')  # Etiqueta para longitud de placas

# ====================================================
# TRAYECTORIA DEL ELECTRÓN
# ====================================================
x, y = -1, 0  # Posición inicial dentro de las placas
v = 3         # Velocidad inicial
F = 1         # Magnitud fuerza magnética

# Fuerza magnética (hacia abajo)
plt.quiver(x, y, 0, -F, angles='xy', scale_units='xy', scale=1,
           linewidth=2, label='F (magnetic)', color='gray')

# Velocidad inicial (con ligera deflexión)
plt.quiver(x, y, +v, -0.2, angles='xy', scale_units='xy', scale=1,
           label='v', color='#CCCCCC')
plt.scatter(x, y, s=100, marker='o', color='black')

# Electrón entrante (antes de placas)
plt.quiver(-10, y, +v, 0, color='#CCCCCC',
           angles='xy', scale_units='xy', scale=1)
plt.scatter(-10, y, s=100, marker='o', color='black')
plt.text(-10, y+0.5, r'$v = E/B$')  # Velocidad inicial

# Electrón saliente (después de placas - trayectoria curva)
y = -1
plt.quiver(9, y, +v, -0.75, color='#CCCCCC',
           angles='xy', scale_units='xy', scale=1)
plt.scatter(9, y, s=100, marker='o', color='black')
plt.text(10, y, r'$v_f$')  # Velocidad final (con deflexión)

# ====================================================
# CONFIGURACIÓN FINAL
# ====================================================
plt.xlim((-13, 15))
plt.ylim((-3, 5))
plt.xticks(())  # Eliminar ejes para diagrama esquemático
plt.yticks(())
plt.legend()
plt.tight_layout()
plt.savefig('experimento_thomson.pdf')
plt.show()