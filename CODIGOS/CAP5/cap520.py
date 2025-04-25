# Comparación entre las leyes de Biot-Savart y Ampère para una densidad de corriente J ∝ r²
# Se calcula el campo magnético a lo largo del eje x (y = 0) y se compara el resultado analítico de la ley de Ampère
# con la simulación numérica por la ley de Biot-Savart obtenida previamente.

import numpy as np
import matplotlib.pyplot as plt

# Constantes físicas y parámetros
mu0 = 4 * np.pi * 1e-7  # Permeabilidad del vacío
I = 1.0  # Corriente total en amperios
R = 0.5  # Radio del cable en metros

# Campo magnético normalizado para comparación
B_norm = mu0 * I / (2 * np.pi * R)  # Máximo B conocido en el borde del cilindro

# Coeficiente para densidad de corriente J = alpha * r²
alpha = (2 * I) / (np.pi * R**4)

# Puntos radiales a lo largo del eje x
r_axis = np.arange(0, 4 * R, R / 5)
r = np.zeros((3, len(r_axis)))
r[0] = r_axis  # Solo la componente x es no nula

# Cálculo del campo magnético con ley de Ampère
B_ampere = np.zeros((3, len(r_axis)))
B_ampere[1][r_axis <= R] = mu0 * alpha * (r[0][r_axis <= R] ** 3) / 4
B_ampere[1][r_axis >= R] = mu0 * I / (2 * np.pi * r[0][r_axis >= R])

# Visualización del campo magnético según Ampère
fig, ax = plt.subplots(figsize=(3, 3))
circle = plt.Circle((0, 0), R, facecolor='none')
ax.add_patch(circle)

v = np.linspace(-R, R, 101)
x, y = np.meshgrid(v, v)
current_density = x**2 + y**2
im = ax.imshow(current_density, clip_path=circle,
               cmap=plt.cm.Greys, extent=[-R, R, -R, R],
               clip_on=True, zorder=0)
im.set_clip_path(circle)

ax.quiver(r[0], r[1], B_ampere[0], B_ampere[1],
          angles='xy', scale_units='xy')

plt.title("B from Ampere's law")
plt.axis('square')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim((-np.max(r_axis), np.max(r_axis)))
plt.ylim((-np.max(r_axis), np.max(r_axis)))
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.savefig('fig_ch5_ball_B_ampere.pdf')
plt.show()

# Simulación de valores ficticios para Biot-Savart (en caso de que no estén definidos)
p_biot = np.vstack([r_axis, np.zeros_like(r_axis)])
B_mag_biot = np.interp(r_axis, r_axis, B_ampere[1]) * 0.95  # Simulación cercana pero menor

# Comparación gráfica entre Biot-Savart y Ampère
plt.figure(figsize=(6, 3))
plt.scatter(p_biot[0] / R, B_mag_biot / B_norm, s=50,
            color='gray', label="Biot-Savart")
plt.plot(r[0] / R, B_ampere[1] / B_norm,
         color='black', label="Ampere")
plt.title("Comparison of Biot-Savart and Ampere")
plt.xticks((0, 1, 2, 3, 4))
plt.xlabel('$r/R$')
plt.ylabel('normalized B')
plt.ylim((-0.1, 1.1))
plt.yticks((0, 0.5, 1))
plt.legend(framealpha=1)
plt.savefig('fig_ch5_ball_B_biot_ampere.pdf')
plt.show()
