import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# Constantes físicas
mu0 = 4 * np.pi * 1e-7  # Permeabilidad magnética del vacío

# Parámetros del sistema
L = 1.0         # Longitud total del alambre
dL = 0.01       # Segmento diferencial
I1 = 1.0        # Corriente fija en el primer alambre
x01, y01 = -0.5, 0.0  # Posición del primer alambre
x02, y02 = 0.5, 0.0   # Posición del segundo alambre

# Función para generar segmentos de corriente
def currents_along_line(I, L, dL, x0=0, y0=0):
    """Genera segmentos discretos de corriente a lo largo de una línea"""
    N = int(L/dL) + 1  # Número de segmentos
    x = np.linspace(-L/2, L/2, N) + x0
    y = np.zeros(N) + y0
    points = np.vstack([x, y])
    current = np.vstack([np.full(N, I/N), np.zeros(N)])
    return points, current

# Función para calcular el campo magnético (versión corregida)
def get_magnetic_field(p_path, p_curr, curr):
    """Calcula B en puntos del camino debido a la distribución de corriente"""
    Bx = np.zeros(p_path.shape[1])
    By = np.zeros(p_path.shape[1])
    
    for i in range(p_curr.shape[1]):
        # Vector desde el segmento de corriente al punto de observación
        rx = p_path[0] - p_curr[0,i]
        ry = p_path[1] - p_curr[1,i]
        r_mag = np.sqrt(rx**2 + ry**2)
        
        # Ley de Biot-Savart (versión diferencial corregida)
        dl = np.array([dL, 0])  # Segmento diferencial en dirección x
        cross_z = dl[0]*ry - dl[1]*rx  # Componente z del producto cruz
        dB_mag = (mu0 / (4 * np.pi)) * (curr[0,i] * cross_z) / (r_mag**3)
        
        # Componentes del campo
        Bx += -dB_mag * (ry / r_mag)  # Componente x de B
        By += dB_mag * (rx / r_mag)   # Componente y de B
    
    return np.array([Bx, By])

# Función para calcular integrales de línea
def line_integral_vector_field(vfield, p):
    """Calcula la integral de línea usando el método del trapecio"""
    dx = np.diff(p[0])
    dy = np.diff(p[1])
    
    # Promedio de los valores del campo en los extremos
    vx_avg = (vfield[0][:-1] + vfield[0][1:]) / 2
    vy_avg = (vfield[1][:-1] + vfield[1][1:]) / 2
    
    return np.sum(vx_avg * dx + vy_avg * dy)

# Generación de caminos cerrados (cuadrados corregidos)
def generate_square_path(center, size, n=100):
    """Genera un camino cuadrado cerrado"""
    x0, y0 = center
    half = size / 2
    top = np.linspace([x0 - half, y0 + half], [x0 + half, y0 + half], n)
    right = np.linspace([x0 + half, y0 + half], [x0 + half, y0 - half], n)
    bottom = np.linspace([x0 + half, y0 - half], [x0 - half, y0 - half], n)
    left = np.linspace([x0 - half, y0 - half], [x0 - half, y0 + half], n)
    return np.hstack([top.T, right.T, bottom.T, left.T])

# Definición de caminos cerrados
C1 = generate_square_path(center=(-0.5, 0.0), size=0.2)  # Alrededor de I1
C2 = generate_square_path(center=(0.5, 0.0), size=0.2)   # Alrededor de I2
C3 = generate_square_path(center=(0.0, 0.0), size=0.2)   # No encierra corrientes
C4 = generate_square_path(center=(0.0, 0.5), size=0.2)   # No encierra corrientes
C5 = generate_square_path(center=(0.0, 0.0), size=1.5)   # Encierra ambas corrientes

# Rango de valores para I2
I2_range = np.linspace(-2.5, 0.5, 13)  # 13 puntos entre -2.5 y 0.5
integral = np.zeros((len(I2_range), 5))

# Pre-cálculo de B1 (solo depende de I1)
p_curr1, curr1 = currents_along_line(I1, L, dL, x0=x01, y0=y01)
B1_C1 = get_magnetic_field(C1, p_curr1, curr1)
B1_C2 = get_magnetic_field(C2, p_curr1, curr1)
B1_C3 = get_magnetic_field(C3, p_curr1, curr1)
B1_C4 = get_magnetic_field(C4, p_curr1, curr1)
B1_C5 = get_magnetic_field(C5, p_curr1, curr1)

# Bucle principal sobre I2
for i, I2 in enumerate(I2_range):
    p_curr2, curr2 = currents_along_line(I2, L, dL, x0=x02, y0=y02)
    
    # Campos B2 para cada camino
    B2_C1 = get_magnetic_field(C1, p_curr2, curr2)
    B2_C2 = get_magnetic_field(C2, p_curr2, curr2)
    B2_C3 = get_magnetic_field(C3, p_curr2, curr2)
    B2_C4 = get_magnetic_field(C4, p_curr2, curr2)
    B2_C5 = get_magnetic_field(C5, p_curr2, curr2)
    
    # Integrales de línea normalizadas
    integral[i, 0] = line_integral_vector_field(B1_C1 + B2_C1, C1) / mu0
    integral[i, 1] = line_integral_vector_field(B1_C2 + B2_C2, C2) / mu0
    integral[i, 2] = line_integral_vector_field(B1_C3 + B2_C3, C3) / mu0
    integral[i, 3] = line_integral_vector_field(B1_C4 + B2_C4, C4) / mu0
    integral[i, 4] = line_integral_vector_field(B1_C5 + B2_C5, C5) / mu0

# Configuración de estilo (corregido)
plt.style.use('default')  # Usamos el estilo por defecto que siempre está disponible

# Primera figura (C1-C4)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(I2_range, integral[:, 0], color='gray', marker='o', label='$C_1$ (around $I_1$)', s=80)
ax.scatter(I2_range, integral[:, 1], color='blue', marker='^', label='$C_2$ (around $I_2$)', s=80)
ax.scatter(I2_range, integral[:, 2], color='green', marker='s', label='$C_3$ (no currents)', s=80)
ax.scatter(I2_range, integral[:, 3], color='red', marker='d', label='$C_4$ (no currents)', s=80)

ax.set_title(r'Line Integral $\frac{1}{\mu_0}\oint \mathbf{B}\cdot d\mathbf{l}$ for Different Paths', pad=20)
ax.set_xlabel('$I_2$ (A)')
ax.set_ylabel(r'$\frac{1}{\mu_0}\oint \mathbf{B}\cdot d\mathbf{l}$ (A)')
ax.axhline(y=I1, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('line_integrals_C1-C4.pdf', bbox_inches='tight')

# Segunda figura (C5)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(I1 + I2_range, integral[:, 4], color='purple', label='$C_5$ (both currents)')
ax.plot([-1.5, 1.5], [-1.5, 1.5], 'k--', label='Expected $I_1 + I_2$')

ax.set_title(r'Line Integral $\frac{1}{\mu_0}\oint \mathbf{B}\cdot d\mathbf{l}$ for Path $C_5$', pad=20)
ax.set_xlabel('$I_1 + I_2$ (A)')
ax.set_ylabel(r'$\frac{1}{\mu_0}\oint \mathbf{B}\cdot d\mathbf{l}$ (A)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('line_integral_C5.pdf', bbox_inches='tight')

plt.show()