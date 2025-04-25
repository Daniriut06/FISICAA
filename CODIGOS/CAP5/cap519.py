# Visualización del campo magnético generado por una densidad de corriente no uniforme J ∝ r²
# utilizando distintos métodos de muestreo (polar, cartesiano, aleatorio).

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
def r2_curr_from_p_curr(I, R, L, p_curr, dV):
    r = np.sqrt(p_curr[0]**2 + p_curr[1]**2)
    N = len(r)
    alpha = (2*I)/(np.pi*(R**4))
    curr = np.zeros((3, N))
    curr[2] = alpha * (r**2) * dV
    return curr

# Muestreo aleatorio dentro del cilindro
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

# Muestreo cartesiano dentro del cilindro
def curr_density_r2_cartesian(I, R, L, dx=0.002):
    x = np.arange(-R, R+dx, dx)
    y = np.arange(-R, R+dx, dx)
    z = np.arange(-L/2, L/2+dx, dx)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    mask = X**2 + Y**2 <= R**2
    x_flat = X[mask].flatten()
    y_flat = Y[mask].flatten()
    z_flat = Z[mask].flatten()
    p_curr = np.vstack((x_flat, y_flat, z_flat))
    dV = dx**3
    curr = r2_curr_from_p_curr(I, R, L, p_curr, dV)
    return p_curr, curr

# Muestreo polar dentro del cilindro
def curr_density_r2_polar(I, R, L, Nr=40, Nphi=40, Nz=40):
    r = np.linspace(0, R, Nr)
    phi = np.linspace(0, 2*np.pi, Nphi)
    z = np.linspace(-L/2, L/2, Nz)
    Rg, Phig, Zg = np.meshgrid(r, phi, z, indexing='ij')
    X = Rg * np.cos(Phig)
    Y = Rg * np.sin(Phig)
    Z = Zg
    p_curr = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))
    dV = (R/Nr)*(2*np.pi/Nphi)*(L/Nz) * R
    curr = r2_curr_from_p_curr(I, R, L, p_curr, dV)
    return p_curr, curr

# Campo magnético en plano xy (ignorando z)
def get_magnetic_field(p_path, p_curr, curr):
    Bx = np.zeros(p_path.shape[1])
    By = np.zeros(p_path.shape[1])
    for i in range(p_curr.shape[1]):
        r0 = p_curr[:, i]
        dl = np.array([0, 0, 1e-3])  # dl ficticio en z
        r = p_path[:2].T - r0[:2]
        r_mag = np.linalg.norm(r, axis=1)
        r_hat = r / r_mag[:, np.newaxis]
        dl_cross_rhat = np.cross(dl, np.hstack((r_hat, np.zeros((len(r_hat), 1)))))[:, 2]
        B_mag = mu0 / (4 * np.pi) * curr[2, i] * dl_cross_rhat / (r_mag ** 2)
        Bx += B_mag * (-r_hat[:, 1])
        By += B_mag * r_hat[:, 0]
    return np.array([Bx, By])

# Cálculo de la integral de línea del campo vectorial
def line_integral_vector_field(vfield, p):
    dx, dy = np.diff(p[0]), np.diff(p[1])
    vx, vy = vfield[0][:-1], vfield[1][:-1]
    val0 = np.sum(vx * dx + vy * dy)
    vx, vy = vfield[0][1:], vfield[1][1:]
    val1 = np.sum(vx * dx + vy * dy)
    return (val0 + val1) / 2

# Genera camino cuadrado cerrado alrededor de un punto dado
def generate_square_path(center, size, n=100):
    x0, y0 = center
    half = size / 2
    top = np.linspace([x0 - half, y0 + half], [x0 + half, y0 + half], n)
    right = np.linspace([x0 + half, y0 + half], [x0 + half, y0 - half], n)
    bottom = np.linspace([x0 + half, y0 - half], [x0 - half, y0 - half], n)
    left = np.linspace([x0 - half, y0 - half], [x0 - half, y0 + half], n)
    path = np.vstack([top, right, bottom, left, top[:1]])
    return path.T

# Función para visualizar el campo B con diferentes métodos de muestreo
def plot_ball_and_B(R, p_curr, curr, rotate=False, title=''):
    p_axis = np.arange(0, 4*R, R/5)
    maxval = np.max(p_axis)
    p = np.zeros((3, len(p_axis)))
    p[0] = p_axis

    B = get_magnetic_field(p, p_curr, curr)
    B_mag = np.sqrt(np.sum(B**2, axis=0))

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

    ax.quiver(p[0], p[1], B[0], B[1], angles='xy', scale_units='xy')

    if rotate:
        phi_range = np.arange(np.pi/4, np.pi*2, np.pi/4)
        for phi in phi_range:
            rot_px = np.cos(phi)*p[0] - np.sin(phi)*p[1]
            rot_py = np.sin(phi)*p[0] + np.cos(phi)*p[1]
            rot_Bx = np.cos(phi)*B[0] - np.sin(phi)*B[1]
            rot_By = np.sin(phi)*B[0] + np.cos(phi)*B[1]
            ax.quiver(rot_px, rot_py, rot_Bx, rot_By,
                      angles='xy', scale_units='xy')

    N = p_curr.shape[1]
    plt.title(r'B con $J \sim r^{2}$ (N=%d)' % N)
    plt.axis('square')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((-maxval, maxval))
    plt.ylim((-maxval, maxval))
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout()
    plt.savefig('fig_ch5_ball_B_%s.pdf' % (title))
    plt.show()

    return p, B_mag

# Ejecutar los 4 métodos de muestreo
p_curr, curr = curr_density_r2_polar(I=1, R=R, L=L, Nr=26, Nphi=32, Nz=50)  # Total ~41664 puntos
_, _ = plot_ball_and_B(R, p_curr, curr, title='polar_41664')

p_curr, curr = curr_density_r2_cartesian(I=1, R=R, L=L, dx=0.005)  # Aproximadamente ~38719 puntos
_, _ = plot_ball_and_B(R, p_curr, curr, title='cartesian_38719')

p_curr, curr = curr_density_r2_random(I=1, R=R, L=L, N=40000)
_, _ = plot_ball_and_B(R, p_curr, curr, title='random_40000')

p_curr, curr = curr_density_r2_random(I=1, R=R, L=L, N=10000000)
p_biot, B_mag_biot = plot_ball_and_B(R, p_curr, curr, title='random_10e7')



