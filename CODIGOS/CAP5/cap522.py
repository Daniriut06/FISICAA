# Cálculo y visualización del rotor (curl) de un campo vectorial en 2 ejemplos.

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from mpl_toolkits.mplot3d import Axes3D

# Variables simbólicas
x, y, z = sym.symbols('x y z')

# Paso de malla para visualización
step = 0.25
v = np.arange(-1, 1 + step, step)
x_range, y_range, z_range = np.meshgrid(v, v, [0], indexing='ij')
p = np.vstack((x_range.flatten(), y_range.flatten(), z_range.flatten()))
N = p.shape[1]

# Trazo de una trayectoria cuadrada para orientación visual
L = 2
dL = step
phi = np.arange(0, 4*L, dL)
px = np.concatenate((np.full(L, -1), np.linspace(-1, 1, L), np.full(L, 1), np.linspace(1, -1, L)))
py = np.concatenate((np.linspace(-1, 1, L), np.full(L, 1), np.linspace(1, -1, L), np.full(L, -1)))
p_line = np.vstack((px, py))

# Dos ejemplos de campos vectoriales para mostrar sus curls
def compute_and_plot_curl(vx, vy, vz, v_line, example_number):
    # Cálculo simbólico del rotor (curl)
    curlx = sym.diff(vz, y) - sym.diff(vy, z)
    curly = sym.diff(vx, z) - sym.diff(vz, x)
    curlz = sym.diff(vy, x) - sym.diff(vx, y)

    print(f"\nExample {example_number}")
    print("\tVector Field:", [vx, vy, vz])
    print("\tCurl:", [curlx, curly, curlz])

    # Evaluación numérica del campo y su curl
    v = np.zeros((3, N))
    curl = np.zeros((3, N))
    for i in range(N):
        substitutions = [(x, p[0, i]), (y, p[1, i]), (z, p[2, i])]
        v[0, i] = vx.subs(substitutions)
        v[1, i] = vy.subs(substitutions)
        v[2, i] = vz.subs(substitutions)
        curl[0, i] = curlx.subs(substitutions)
        curl[1, i] = curly.subs(substitutions)
        curl[2, i] = curlz.subs(substitutions)

    # Gráficas
    fig = plt.figure(figsize=(6, 3))
    grid = fig.add_gridspec(1, 2)

    ax0 = fig.add_subplot(grid[0, 0])
    ax0.quiver(p[0], p[1], v[0], v[1], scale=5, color='gray')
    ax0.plot(p_line[0], p_line[1], 'r--', alpha=0.3)  # trayectoria visual
    ax0.axis('square')
    ax0.set_xlim([-1.5, 1.5])
    ax0.set_ylim([-1.5, 1.5])
    ax0.set_xticks([-1, 0, 1])
    ax0.set_yticks([-1, 0, 1])
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_title('Vector Field')

    ax1 = fig.add_subplot(grid[0, 1], projection='3d')
    ax1.quiver(p[0], p[1], p[2], curl[0], curl[1], curl[2], length=0.2, color='black')
    ax1.set_xlim([-1.25, 1.25])
    ax1.set_ylim([-1.25, 1.25])
    ax1.set_zlim([-1, 1])
    ax1.set_xticks([-1, 0, 1])
    ax1.set_yticks([-1, 0, 1])
    ax1.set_zticks([-1, 0, 1])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Curl')

    plt.tight_layout()
    plt.savefig(f'fig_ch5_curl_ex{example_number}.pdf', bbox_inches='tight')
    plt.show()

# Primer ejemplo: campo rotacional (tipo rotación)
vx1 = -y
vy1 = x
vz1 = sym.Integer(0)
v_line1 = np.vstack((-p_line[1], p_line[0]))
compute_and_plot_curl(vx1, vy1, vz1, v_line1, example_number=1)

# Segundo ejemplo: campo no lineax2 = -y**2
vy2 = x**2
vz2 = sym.Integer(0)
v_line2 = np.vstack((-p_line[1]**2, p_line[0]**2))
compute_and_plot_curl(vx2, vy2, vz2, v_line2, example_number=2)
