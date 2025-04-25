# ================================================================
# Este código genera una figura que compara cómo la orientación de 
# una superficie afecta el flujo eléctrico que atraviesa dicha superficie.
# En el caso (a), la superficie es perpendicular al campo eléctrico.
# En el caso (b), la superficie está inclinada (slanted).
# Las líneas negras representan el flujo total y las flechas muestran
# el campo eléctrico generado por una fuente en el origen.
# ================================================================

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del dibujo
lw = 6             # Grosor de la línea negra (flujo)
scale = 0.2        # Escala de las flechas de campo eléctrico
dphi = 0.2         # Separación entre las líneas grises guía (ángulo sólido)
loc = 7            # Posición x donde se coloca la superficie

# Crear figura con 2 subgráficos en 1 columna
fig = plt.figure(figsize=(6, 4))
gs = fig.add_gridspec(2, 1)  # Grid de 2 filas, 1 columna

# Títulos para los dos subgráficos
ax_titles = ['(a) Perpendicular', '(b) Slanted']

# Generar cada subfigura
for i in range(2):
    ax = fig.add_subplot(gs[i])  # Añadir subplot
    ax.set_title(ax_titles[i])   # Título correspondiente

    # Dibujar el origen (la fuente) con '+'
    ax.scatter(0, 0, marker='+', color='black')

    # Líneas guía en ángulo (como conos de flujo)
    ax.plot([0, 10], [0, +dphi], color='gray', linewidth=1)
    ax.plot([0, 10], [0, -dphi], color='gray', linewidth=1)

    # Control de inclinación: 0 para perpendicular, 0.7 para slanted
    xs = i * 0.7

    # Línea negra que representa el flujo total a través de la superficie
    ax.plot(
        [loc + xs, loc - xs],
        np.array([-1, 1]) * dphi * loc / 10,
        color='black',
        linewidth=lw,
        alpha=0.5
    )

    # Malla de puntos para dibujar las flechas del campo eléctrico
    x, y = np.meshgrid(
        np.array([loc - 0.8, loc, loc + 0.8]),
        np.arange(-0.3, 0.4, 0.15),
        indexing='ij'
    )

    # Flechas del campo eléctrico apuntando desde el origen
    ax.quiver(
        x, y,
        x / (x**2 + y**2), y / (x**2 + y**2),  # componentes normalizadas
        angles='xy',
        scale_units='xy',
        scale=scale,
        color='gray',
        linewidth=1
    )

    # Configuración visual del subgráfico
    ax.set_xlim((-1, 11))
    ax.set_ylim((-0.5, 0.5))
    ax.axis('off')  # Quitar ejes para mayor claridad

# Ajustar diseño y guardar la figura
plt.tight_layout()
plt.savefig('fig_ch4_gauss_boundary_orientation.pdf')
plt.show()
