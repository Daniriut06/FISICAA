""# Código 5.14 modificado - reemplazando I2 con distribución cilíndrica no uniforme (J ∝ r²)

import numpy as np
import matplotlib.pyplot as plt

mu0 = 4 * np.pi * 1e-7

# Parámetros del alambre
L = 1.0
R = 0.05  # Radio del cilindro para corriente I2

# Posiciones iniciales
x01, y01 = -0.5, 0.0  # Corriente I1
x02, y02 = 0.5, 0.0   # Corriente I2 (cilindro)

# Corriente I1 lineal, como antes
def currents_along_line(I, L, dL, x0=0, y0=0):
    x = np.arange(-L/2, L/2 + dL, dL)
    y = np.zeros_like(x)
    p = np.vstack([x + x0, y + y0])
    current = np.vstack([np.full_like(x, I / len(x)), np.zeros_like(x)])
    return p, current

# Distribución de corriente J = α r² dentro de un cilindro
# Versión random sampling, más rápida

def r2_curr_from_p_curr(I, R, L, p_curr, dV):
    r = np.sqrt(p_curr[0]**2 + p_curr[1]**2)
    N = len(r)
    alpha = (2*I)/(np.pi*(R**4))
    curr = np.zeros((3, N))
    curr[2] = alpha * (r**2) * dV
    return curr

def curr_density_r2_random(I, R, L, N=5000, x0=0, y0=0):
    r = np.sqrt(np.random.rand(N)) * R
    phi = np.random.rand(N) * 2 * np.pi
    z = (np.random.rand(N)*2 - 1) * L/2
    x = r * np.cos(phi) + x0
    y = r * np.sin(phi) + y0
    dV = np.pi * (R**2) * L / N
    p_curr = np.vstack((x, y, z))
    curr = r2_curr_from_p_curr(I, R, L, p_curr, dV)
    return p_curr, curr

# Campo magnético en plano xy (ignorando z)
def get_magnetic_field(p_path, p_curr, curr):
    Bx = np.zeros(p_path.shape[1])
    By = np.zeros(p_path.shape[1])
    for i in range(p_curr.shape[1]):
        r0 = p_curr[:, i]
        dl = np.array([0, 0, 1e-3])  # dl ficticio en z
        r = p_path.T - r0[:2]
        r_mag = np.linalg.norm(r, axis=1)
        r_hat = r / r_mag[:, np.newaxis]
        dl_cross_rhat = np.cross(dl, np.hstack((r_hat, np.zeros((len(r_hat), 1)))))[:, 2]
        B_mag = mu0 / (4 * np.pi) * curr[2, i] * dl_cross_rhat / (r_mag ** 2)
        Bx += B_mag * (-r_hat[:, 1])
        By += B_mag * r_hat[:, 0]
    return np.array([Bx, By])

def line_integral_vector_field(vfield, p):
    dx, dy = np.diff(p[0]), np.diff(p[1])
    vx, vy = vfield[0][:-1], vfield[1][:-1]
    val0 = np.sum(vx * dx + vy * dy)
    vx, vy = vfield[0][1:], vfield[1][1:]
    val1 = np.sum(vx * dx + vy * dy)
    return (val0 + val1) / 2

def generate_square_path(center, size, n=100):
    x0, y0 = center
    half = size / 2
    top = np.linspace([x0 - half, y0 + half], [x0 + half, y0 + half], n)
    right = np.linspace([x0 + half, y0 + half], [x0 + half, y0 - half], n)
    bottom = np.linspace([x0 + half, y0 - half], [x0 - half, y0 - half], n)
    left = np.linspace([x0 - half, y0 - half], [x0 - half, y0 + half], n)
    path = np.vstack([top, right, bottom, left, top[:1]])
    return path.T

C1 = generate_square_path(center=(-0.5, 0.0), size=0.2)
C2 = generate_square_path(center=(0.5, 0.0), size=0.2)
C3 = generate_square_path(center=(0.0, 0.0), size=0.2)
C4 = generate_square_path(center=(0.0, 0.5), size=0.2)
C5 = generate_square_path(center=(0.0, 0.0), size=1.5)

I1 = 1
I2_range = np.arange(-2.5, 0.6, 0.25)
integral = np.zeros((len(I2_range), 5))

# Corriente I1 (lineal)
p_curr1, curr1 = currents_along_line(I1, L, 0.01, x0=x01, y0=y01)
B1_C1 = get_magnetic_field(C1, p_curr1, curr1)
B1_C2 = get_magnetic_field(C2, p_curr1, curr1)
B1_C3 = get_magnetic_field(C3, p_curr1, curr1)
B1_C4 = get_magnetic_field(C4, p_curr1, curr1)
B1_C5 = get_magnetic_field(C5, p_curr1, curr1)

for i, I2 in enumerate(I2_range):
    p_curr2, curr2 = curr_density_r2_random(I2, R, L, N=3000, x0=x02, y0=y02)
    B2_C1 = get_magnetic_field(C1, p_curr2, curr2)
    B2_C2 = get_magnetic_field(C2, p_curr2, curr2)
    B2_C3 = get_magnetic_field(C3, p_curr2, curr2)
    B2_C4 = get_magnetic_field(C4, p_curr2, curr2)
    B2_C5 = get_magnetic_field(C5, p_curr2, curr2)

    lint_C1 = line_integral_vector_field(B1_C1 + B2_C1, C1)
    lint_C2 = line_integral_vector_field(B1_C2 + B2_C2, C2)
    lint_C3 = line_integral_vector_field(B1_C3 + B2_C3, C3)
    lint_C4 = line_integral_vector_field(B1_C4 + B2_C4, C4)
    lint_C5 = line_integral_vector_field(B1_C5 + B2_C5, C5)

    integral[i, :] = np.array([lint_C1, lint_C2, lint_C3, lint_C4, lint_C5]) / mu0

fig = plt.figure(figsize=(4, 3))
plt.scatter(I2_range, integral[:, 0], color='#CCCCCC', label='$C_1$', s=90)
plt.scatter(I2_range, integral[:, 1], color='black', label='$C_2$', s=10)
plt.scatter(I2_range, integral[:, 2], color='#808080', label='$C_3$', s=90)
plt.scatter(I2_range, integral[:, 3], color='#EEEEEE', label='$C_4$', s=10)
plt.title(r'$\oint B\,dl$ over $C_1, C_2, C_3, C_4$')
plt.xlabel('$I_2$')
plt.ylabel(r'$\frac{1}{\mu_0}\oint B\,dl$')
plt.xticks([-2, -1, 0])
plt.yticks([0, 0.5, 1])
plt.legend(framealpha=1)
plt.savefig('fig_ch5_diff_paths_r2_1.pdf', bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(4, 3))
plt.scatter(I1 + I2_range, integral[:, 4], label='$C_5$', color='black')
plt.plot([-1.5, 1.5], [-1.5, 1.5], color='#AAAAAA', label='Line with a slope of 1')
plt.title(r'$\oint B\,dl$ over $C_5$')
plt.xlabel('$I_1 + I_2$')
plt.ylabel(r'$\frac{1}{\mu_0}\oint B\,dl$')
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.legend(framealpha=1)
plt.savefig('fig_ch5_diff_paths_r2_2.pdf', bbox_inches='tight')
plt.show()
