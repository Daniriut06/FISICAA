# Este código calcula y visualiza el campo magnético producido por dos hilos largos con corrientes opuestas.
# También dibuja caminos cerrados (cuadrados y círculos) para análisis de la ley de Ampère.

import numpy as np
import matplotlib.pyplot as plt

# ========== Funciones de corriente y campo ==========

def currents_along_line(I, L, dL, x0=0, y0=0):
    z = np.linspace(-L/2, L/2, int(L/dL))
    p_curr = np.array([[x0]*len(z), [y0]*len(z), z])
    curr = np.array([I]*len(z))
    return p_curr, curr

def get_magnetic_field(p, p_curr, curr, dl=0.1):
    mu0 = 1
    B = np.zeros((3, p.shape[1]))

    for i in range(p_curr.shape[1] - 1):
        r0 = p_curr[:, i]
        r1 = p_curr[:, i+1]
        dl_vec = r1 - r0
        mid = 0.5 * (r0 + r1)
        I_dl = curr[i] * dl_vec

        r_vec = p.T - mid
        r_mag = np.linalg.norm(r_vec, axis=1)
        r_mag[r_mag < 1e-5] = 1e-5  # evitar divisiones por cero
        r_hat = r_vec / r_mag[:, np.newaxis]

        contrib = np.cross(I_dl, r_hat)
        contrib = (mu0 / (4 * np.pi)) * contrib.T / (r_mag ** 2)
        B += contrib

    return B

def points_cartesian_xy(xmin, xmax, ymin, ymax, delta):
    x = np.arange(xmin, xmax+delta, delta)
    y = np.arange(ymin, ymax+delta, delta)
    X, Y = np.meshgrid(x, y)
    return np.vstack([X.ravel(), Y.ravel()]), X, Y

def path_square(xc, yc, side, delta=0.01):
    L = side
    p = []
    for x in np.arange(-L/2, L/2, delta):
        p.append([x + xc, -L/2 + yc])
    for y in np.arange(-L/2, L/2, delta):
        p.append([L/2 + xc, y + yc])
    for x in np.arange(L/2, -L/2, -delta):
        p.append([x + xc, L/2 + yc])
    for y in np.arange(L/2, -L/2, -delta):
        p.append([-L/2 + xc, y + yc])
    return np.array(p).T

def currents_along_circle(I, R, dtheta):
    theta = np.arange(0, 2*np.pi, dtheta)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    p_curr = np.vstack((x, y, np.zeros_like(x)))
    curr = np.array([I]*len(theta))
    return p_curr, curr

# ========== Parámetros de los hilos ==========

L, dL = 100, 0.1
I1, x01, y01 = +1, -0.65, -0.15
I2, x02, y02 = -1, +0.65, +0.15

p_curr1, curr1 = currents_along_line(I1, L, dL, x0=x01, y0=y01)
p_curr2, curr2 = currents_along_line(I2, L, dL, x0=x02, y0=y02)

# ========== Malla para el campo ==========

xy, X, Y = points_cartesian_xy(-2, 2, -2, 2, 0.25)
z = np.zeros((1, xy.shape[1]))
p = np.vstack([xy, z])

B1 = get_magnetic_field(p, p_curr1, curr1, dl=dL)
B2 = get_magnetic_field(p, p_curr2, curr2, dl=dL)
B = B1 + B2

# ========== Caminos cerrados ==========

C1 = path_square(xc=-0.25, yc=0.35, side=0.5)
C2 = path_square(xc=-0.5, yc=-0.5, side=1)
C3 = path_square(xc=1.0, yc=1.0, side=0.75)
C4, _ = currents_along_circle(0, R=0.5, dtheta=np.pi/100)
C4[0] += 1.5
C5, _ = currents_along_circle(0, R=2, dtheta=np.pi/100)

# ========== Gráfica ==========

plt.figure(figsize=(6, 6))

# Dibujar los hilos
def get_marker(I): return '.' if I > 0 else 'x'

plt.scatter(p_curr1[0, 0], p_curr1[1, 0], color='gray', s=200)
plt.scatter(p_curr2[0, 0], p_curr2[1, 0], color='gray', s=200)
plt.scatter(p_curr1[0, 0], p_curr1[1, 0], color='white', s=80, marker=get_marker(I1))
plt.scatter(p_curr2[0, 0], p_curr2[1, 0], color='white', s=80, marker=get_marker(I2))

# Normalizar los vectores para evitar saturación
B_norm = np.sqrt(B[0]**2 + B[1]**2)
B[0] = B[0] / B_norm
B[1] = B[1] / B_norm

# Dibujar el campo magnético con quiver
plt.quiver(p[0], p[1], B[0], B[1], color='black', angles='xy', scale_units='xy', scale=10)

# Caminos cerrados
for C, label, i in zip([C1, C2, C3, C4, C5], ['C₁', 'C₂', 'C₃', 'C₄', 'C₅'], [24, 100, 0, 50, 100]):
    plt.plot(C[0], C[1], color='gray', linewidth=2)
    plt.text(C[0, i]+0.05, C[1, i]+0.05, f"${label}$", fontsize=14)

# Ajustes de gráfico
plt.xlabel('x')
plt.ylabel('y')
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.xticks([-2, 0, 2])
plt.yticks([-2, 0, 2])
plt.title("Currents, magnetic fields, and closed paths")
plt.axis('square')
plt.tight_layout()
plt.savefig('fig_ch5_curr_B_paths_fixed.pdf', bbox_inches='tight')
plt.show()
