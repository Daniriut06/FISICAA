"""
Código 5.15 - Cálculo del campo magnético total generado por múltiples corrientes paralelas

Este script calcula el campo magnético resultante en un punto central producido por
varias corrientes paralelas equidistantes. Requiere las funciones currents_along_line()
y get_magnetic_field() definidas previamente.
"""

import numpy as np
import matplotlib.pyplot as plt

# Función auxiliar necesaria (debe estar definida previamente)
def currents_along_line(I, L, dL, x0=0, y0=0):
    """Genera segmentos discretos de corriente a lo largo de una línea"""
    N = int(L/dL) + 1
    x = np.linspace(-L/2, L/2, N) + x0
    y = np.zeros(N) + y0
    points = np.vstack([x, y, np.zeros(N)])  # Coordenadas 3D
    current = np.vstack([np.zeros(N), np.full(N, I/N), np.zeros(N)])  # Corriente en dirección y
    return points, current

# Función auxiliar necesaria (debe estar definida previamente)
def get_magnetic_field(p, p_curr, curr):
    """Calcula B en un punto debido a una distribución de corriente"""
    B = np.zeros(3)
    for i in range(p_curr.shape[1]):
        r = p - p_curr[:, i:i+1]
        r_mag = np.linalg.norm(r)
        if r_mag < 1e-10:  # Evitar división por cero
            continue
        dB = (1e-7) * np.cross(curr[:, i], r.T).T / (r_mag**3)
        B += dB.flatten()
    return B.reshape(3, 1)

def calculate_B_multiple_currents(x0_range, I=1, L=10, dL=0.1):
    """
    Calcula el campo magnético total en el origen producido por múltiples corrientes.
    
    Parámetros:
    x0_range : array - Posiciones en x de los conductores
    I : float - Intensidad de corriente
    L : float - Longitud del conductor
    dL : float - Segmento diferencial
    
    Retorna:
    B_tot : array (3,1) - Campo magnético total [Bx, By, Bz]
    """
    p = np.array([[0], [0], [0]])  # Punto de observación
    B_tot = np.zeros((3, 1))
    
    for x0 in x0_range:
        p_curr, curr = currents_along_line(I, L, dL, x0=x0, y0=0)
        B = get_magnetic_field(p, p_curr, curr)
        B_tot += B
    
    return B_tot

def plot_multiple_currents(x0_range):
    """Visualiza la disposición de los conductores"""
    plt.figure(figsize=(5, 1))
    for x0 in x0_range:
        plt.scatter(x0, 0, color='black', s=200)
        plt.scatter(x0, 0, color='white', s=80, marker='.')
    
    plt.ylim((-0.5, 0.5))
    plt.xlim((-1.1, 1.1))
    plt.gca().set_yticks([])
    plt.tight_layout()

# Configuraciones de prueba
configs = [
    {'n': 2, 'dx': 0.2},  # 2 conductores
    {'n': 4, 'dx': 0.2},  # 4 conductores  
    {'n': 6, 'dx': 0.2}   # 6 conductores
]

for cfg in configs:
    x0_range = np.arange(-cfg['n']*cfg['dx']/2, 
                        cfg['n']*cfg['dx']/2 + cfg['dx'], 
                        cfg['dx'])
    
    B_total = calculate_B_multiple_currents(x0_range)
    print(f"{len(x0_range)} conductores - B_total:", B_total.T[0])
    
    plot_multiple_currents(x0_range)
    plt.savefig(f'fig_mult_current_{len(x0_range)}.pdf', bbox_inches='tight')
    plt.show()

# Ejemplo de precisión numérica
print("\nPrecisión de np.arange:", np.arange(-1, 1.1, 0.1))