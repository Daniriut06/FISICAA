# ================================================================
# Este código genera una figura comparando una superficie perpendicular
# con una superficie inclinada respecto al campo eléctrico. Se ilustran:
# - Diferentes orientaciones del vector normal a la superficie,
# - Elementos diferenciales de longitud (dl),
# - El ángulo θ entre ambas configuraciones.
# ================================================================

import numpy as np
import matplotlib.pyplot as plt

# Parámetros gráficos
lw = 12         # Grosor de las líneas principales (superficies)
fs = 14         # Tamaño de fuente para los textos

# Crear figura
fig = plt.figure(figsize=(4, 4))

# --- Superficie perpendicular ---
# Línea vertical en el centro representando dl⊥
plt.plot([+0.0, -0.0], [-0.5, +0.5], color='black', linewidth=lw)

# Vector normal unitario hacia la derecha (campo eléctrico)
plt.quiver(
    0, 0,     # origen
    1, 0,     # dirección (horizontal)
    angles='xy', scale_units='xy', scale=2, color='black'
)

# Etiquetas para la perpendicular
plt.text(-0.15, 0.35, r'$dl_{\perp}$', fontsize=fs)
plt.text(0.5, 0.05, r'$\hat{n}_{\perp}$', fontsize=fs)

# --- Superficie inclinada (slanted) ---
# Línea inclinada representando dl_s (slanted dl)
plt.plot([+0.5, -0.5], [-0.5, +0.5], color='gray', linewidth=lw)

# Vector normal inclinado (también en dirección del campo pero no alineado)
plt.quiver(
    0, 0,         # origen
    0.7, 0.7,     # dirección (diagonal)
    angles='xy', scale_units='xy', scale=2, color='gray'
)

# Etiquetas para la superficie inclinada
plt.text(-0.4, 0.2, r'$dl_{s}$', fontsize=fs)
plt.text(0.3, 0.4, r'$\hat{n}_{s}$', fontsize=fs)

# Líneas punteadas horizontales que marcan el ángulo θ
plt.plot([0, -0.5], [+0.5, +0.5], color='gray', linewidth=1, ls='--')
plt.plot([0, +0.5], [-0.5, -0.5], color='gray', linewidth=1, ls='--')

# Texto para marcar el ángulo θ entre normal perpendicular e inclinada
plt.text(-0.1, 0.15, r'$\theta$', fontsize=fs)
plt.text(0.15, 0.05, r'$\theta$', fontsize=fs)

# Ajustes visuales
plt.axis('square')     # Mantener proporciones iguales
plt.axis('off')        # Ocultar ejes
plt.tight_layout()     # Eliminar espacios innecesarios

# Guardar figura
plt.savefig('fig_ch4_perp_vs_slanted.pdf', bbox_inches='tight')

# Mostrar figura
plt.show()
