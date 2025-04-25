import numpy as np
import matplotlib.pyplot as plt

"""
==========================================================
CÁLCULO DE INTEGRALES DE LÍNEA DE CAMPOS MAGNÉTICOS CERRADOS
==========================================================
Este script calcula la integral de línea del campo magnético producido
por dos corrientes sobre cinco caminos cerrados circulares (C1 a C5),
aplicando la Ley de Ampère en su forma integral.
"""

# --- CONSTANTES ---
mu0 = 4 * np.pi * 1e-7  # permeabilidad del vacío

# --- FUNCIONES ---

def get_magnetic_field(path, current_pos, current_val):
    """
    Calcula el campo magnético producido por una corriente puntual en 2D.
    """
    x, y = path
    x0, y0 = current_pos
    dx = x - x0
    dy = y - y0
    r2 = dx**2 + dy**2
    r2[r2 == 0] = 1e-20  # evitar división por cero
    Bx = -mu0 * current_val * dy / (2 * np.pi * r2)
    By =  mu0 * current_val * dx / (2 * np.pi * r2)
    return (Bx, By)

def line_integral_vector_field(vfield, p):
    """
    Calcula la integral de línea de un campo vectorial a lo largo del contorno p.
    """
    dx, dy = np.diff(p[0]), np.diff(p[1])
    vx0, vy0 = vfield[0][:-1], vfield[1][:-1]
    vx1, vy1 = vfield[0][1:], vfield[1][1:]
    val0 = np.sum(vx0 * dx + vy0 * dy)
    val1 = np.sum(vx1 * dx + vy1 * dy)
    return (val0 + val1) / 2

# --- DEFINIR CORRIENTES ---
p_curr1 = (0.0, 0.0)   # Corriente 1 en el origen
curr1 = 1.0            # 1 A

p_curr2 = (0.03, 0.0)  # Corriente 2 un poco a la derecha
curr2 = -1.0           # -1 A

# --- DEFINIR CAMINOS CERRADOS (CÍRCULOS DE DISTINTOS RADIOS) ---

def create_circle_path(radius, n_points=500):
    theta = np.linspace(0, 2*np.pi, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return (x, y)

radii = [0.01, 0.02, 0.04, 0.06, 0.08]
C1, C2, C3, C4, C5 = [create_circle_path(r) for r in radii]

# --- CALCULAR CAMPOS MAGNÉTICOS EN CADA CAMINO ---

B1_Cs = [get_magnetic_field(C, p_curr1, curr1) for C in [C1, C2, C3, C4, C5]]
B2_Cs = [get_magnetic_field(C, p_curr2, curr2) for C in [C1, C2, C3, C4, C5]]

# --- CALCULAR INTEGRALES DE LÍNEA PARA CADA CAMINO ---

lint = []
for i in range(5):
    Bx_total = B1_Cs[i][0] + B2_Cs[i][0]
    By_total = B1_Cs[i][1] + B2_Cs[i][1]
    lint_val = line_integral_vector_field((Bx_total, By_total), [C1, C2, C3, C4, C5][i])
    lint.append(lint_val)

# --- IMPRIMIR RESULTADOS ---

print("=== Resultados de integrales de línea (dividido por mu0) ===")
for i, v in enumerate(np.array(lint) / mu0):
    print("Line integral over C%d = %+7.4f" % (i + 1, v))

# --- GRAFICAR LAS CURVAS Y CORRIENTES ---
for i, C in enumerate([C1, C2, C3, C4, C5], start=1):
    plt.plot(C[0], C[1], label=f'C{i}')
plt.scatter(*p_curr1, color='r', label='Corriente 1')
plt.scatter(*p_curr2, color='g', label='Corriente 2')
plt.gca().set_aspect('equal')
plt.title("Caminos cerrados y corrientes")
plt.legend()
plt.grid(True)
plt.show()
