# Importación de librerías para cálculos numéricos y visualización 3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================
# GENERACIÓN DE SUPERFICIES CON MISMO BORDE
# ==============================================

# Parámetros del disco (radio y paso de discretización)
R = 0.5       # Radio del disco circular
dR = 0.01     # Incremento para coordenada radial

# Crear malla de coordenadas polares (r, phi) y convertir a cartesianas (x, y)
r, phi, z = np.meshgrid(
    np.arange(0, R, dR),            # Coordenada radial (0 a R)
    np.arange(0, 2*np.pi, np.pi/100),  # Coordenada angular (0 a 2pi)
    0,                              # Coordenada z (fija en 0)
    indexing='ij'                   # Indexación 'ij' para matrices
)
x, y = r*np.cos(phi), r*np.sin(phi)  # Conversión a cartesianas

# Aplanar las coordenadas para la visualización
p = np.vstack((x.flatten(), y.flatten(), z.flatten()))

# ==============================================
# DEFINICIÓN DE 3 SUPERFICIES CON MISMO BORDE
# ==============================================
surface = [
    0*(p[0]+p[1]),  # 1. Plano completamente horizontal (z = 0)
    
    # 2. Superficie ondulada con patrón coseno
    -np.cos((3*np.pi/2)*(p[0]**2+p[1]**2)/(0.5**2)),
    
    # 3. Superficie exponencial (forma de campana invertida)
    (1.5*np.exp(-((p[0]**2+p[1]**2)-0.5**2)/0.3))-1.5
]

# ==============================================
# VISUALIZACIÓN DE LAS SUPERFICIES
# ==============================================
fig = plt.figure(figsize=(15, 5))  # Figura con 3 subplots

for i in range(3):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')  # Subplot 3D
    
    # Graficar superficie triangular
    ax.plot_trisurf(p[0], p[1], surface[i], cmap='gray')
    
    # Configuración de ejes para mejor visualización
    ax.set_xticks([])  # Eliminar ticks en X
    ax.set_yticks([])  # Eliminar ticks en Y
    ax.set_zticks([])  # Eliminar ticks en Z
    
    # Límites fijos para comparación consistente
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-1.5, 0.5])

# Ajuste de layout y guardado de figura
plt.tight_layout()
plt.savefig('superficies_mismo_borde.pdf')  # Guardar como PDF
plt.show()  # Mostrar figura interactiva