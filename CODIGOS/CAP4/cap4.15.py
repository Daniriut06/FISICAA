# ===============================================
# Código para visualizar el flujo eléctrico a diferentes distancias
# Este código genera dos diagramas comparativos: uno cerca y otro lejos
# de una región donde existe una densidad de flujo. Se muestra cómo la
# misma cantidad de flujo se reparte en diferentes regiones dependiendo
# de la distancia, usando líneas de campo eléctrico y un segmento
# representando el flujo total.
# ===============================================

import numpy as np
import matplotlib.pyplot as plt

# Grosor de la línea del flujo total
lw = 6
# Escala para la longitud de las flechas del campo eléctrico
scale = 0.2
# Separación vertical entre las dos líneas grises de guía
dphi = 0.2
# Posiciones x de los dos casos: cerca (5) y lejos (9)
loc = [5, 9]

# Crear la figura con dos filas (una para cada caso)
fig = plt.figure(figsize=(6, 4))
gs = fig.add_gridspec(2, 1)

# Títulos para cada subgráfico
ax_titles = ['(a) near', '(b) far-away']

# Bucle sobre los dos casos: cerca y lejos
for i in range(2):
    ax = fig.add_subplot(gs[i])
    ax.set_title(ax_titles[i])
    
    # Marcar el origen con un símbolo '+'
    ax.scatter(0, 0, marker='+', color='black')

    # Dibujar dos líneas guía que representan la región donde pasa el flujo
    ax.plot([0, 10], [0, +dphi], color='gray', linewidth=1)
    ax.plot([0, 10], [0, -dphi], color='gray', linewidth=1)

    # Dibujar una línea negra vertical representando el "flujo total"
    ax.plot(
        np.array([1, 1]) * loc[i],  # x constante
        np.array([-1, 1]) * dphi * loc[i] / 10,  # altura proporcional
        color='black',
        linewidth=lw,
        alpha=0.25  # transparencia
    )

    # Crear malla para las posiciones donde se dibujarán las flechas del campo
    x, y = np.meshgrid(loc[i], np.arange(-0.3, 0.4, 0.15), indexing='ij')

    # Dibujar vectores del campo eléctrico con dirección radial
    ax.quiver(
        x, y,
        x / (x**2 + y**2), y / (x**2 + y**2),  # componentes del campo normalizado
        angles='xy',
        scale_units='xy',
        scale=scale,
        color='gray',
        linewidth=1
    )

    # Ajustar los límites y quitar ejes
    ax.set_xlim((-1, 11))
    ax.set_ylim((-0.5, 0.5))
    ax.axis('off')

# Acomodar los elementos y guardar el gráfico
plt.tight_layout()
plt.savefig('fig_ch4_gauss_boundary_distance.pdf', bbox_inches='tight')
plt.show()
