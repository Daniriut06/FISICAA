# Calcula y visualiza el rotor (curl) de campos vectoriales simbólicos en 3D.
# Para dos campos vectoriales distintos se obtiene su rotacional y se grafican:
# (1) El campo en 2D (componente xy), (2) El rotacional en 3D.

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# Definir variables simbólicas
x, y, z = sym.symbols('x y z')

# Crear malla de puntos para evaluar
step = 0.25
v = np.arange(-1, 1 + step, step)
x_range, y_range, z_range = np.meshgrid(v, v, 0, indexing='ij')  # plano z = 0
p = np.vstack((x_range.flatten(), y_range.flatten(), z_range.flatten()))
N = p.shape[1]

# Función para calcular integral de línea sobre un campo vectorial
def line_integral_vector_field(vfield, path):
    dr = np.diff(path)
    f_avg = 0.5 * (vfield[:, :-1] + vfield[:, 1:])
    integrand = np.sum(f_avg * dr, axis=0)
    return np.sum(integrand)

# Función para generar un contorno cuadrado
def currents_along_square(I=0, L=1, dL=0.001):
    t = np.arange(0, 4*L, dL)
    x = np.piecewise(t, [t<L, (t>=L)&(t<2*L), (t>=2*L)&(t<3*L), t>=3*L],
                    [lambda t: t, lambda t: L, lambda t: 3*L - t, lambda t: 0])
    y = np.piecewise(t, [t<L, (t>=L)&(t<2*L), (t>=2*L)&(t<3*L), t>=3*L],
                    [lambda t: 0, lambda t: t - L, lambda t: L, lambda t: 4*L - t])
    return np.vstack((x - L/2, y - L/2)), t

# Función para generar un contorno circular
def currents_along_circle(I=0, R=0.5, d_phi=np.pi/128):
    phi = np.arange(0, 2 * np.pi, d_phi)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    path = np.vstack((x, y))
    return path, phi

# Función para graficar y demostrar el teorema de Stokes
def demo_stokes(p_boundary, step=0.001, skip=200, title=''):
    dxdy = step**2
    fig, axs = plt.subplots(2, 3, figsize=(7, 6), sharey=True, sharex=True)
    for s in range(2):
        for ctr in range(3):
            x, y = np.meshgrid(np.arange(-0.5, 0.5, step), np.arange(-0.5, 0.5, step), indexing='ij')
            p_area = np.vstack((x.flatten(), y.flatten()))

            if title == 'circle':
                r = np.sqrt(p_area[0]**2 + p_area[1]**2)
                R = np.max(p_boundary[0])
                p_area = p_area[:, r <= R]

            if ctr == 0:
                x0, y0 = 0, 0
            elif ctr == 1:
                x0, y0 = -0.25, -0.5
            else:
                x0, y0 = 0.5, -0.5

            p_line = np.zeros(p_boundary.shape)
            p_line[0], p_line[1] = p_boundary[0] + x0, p_boundary[1] + y0
            p_area[0], p_area[1] = p_area[0] + x0, p_area[1] + y0

            if s == 0:
                v_line = np.vstack((-p_line[1], p_line[0]))
                curlz = 2 + np.zeros(p_area.shape[1])
            else:
                v_line = np.vstack((-p_line[1]**2, p_line[0]**2))
                curlz = 2 * (p_area[0] + p_area[1])

            method1 = line_integral_vector_field(v_line, p_line)
            method2 = np.sum(curlz) * dxdy

            ax = axs[s, ctr]
            ax.plot(p_line[0], p_line[1], color='black', linewidth=1)
            ax.quiver(p_line[0][::skip], p_line[1][::skip],
                      v_line[0][::skip], v_line[1][::skip],
                      scale=4, color='gray')
            ax.axis('square')
            ax.set_xlim(np.array([-1, 2]))
            ax.set_ylim(np.array([-1, 2]))
            ax.set_xticks([-1, 0, 1, 2])
            ax.set_yticks([-1, 0, 1, 2])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Example %d' % (s + 1))
            ax.legend((r'$\oint_C$: %+4.3f' % method1,
                       r'$\iint_A$: %+4.3f' % method2),
                      handlelength=0)

    plt.tight_layout()
    plt.savefig(f'fig_ch5_stokes_{title}.pdf', bbox_inches='tight')
    plt.show()

# Ejecutar demostración con contorno cuadrado y circular
step = 0.001
p_boundary, _ = currents_along_square(I=0, L=1, dL=step)
demo_stokes(p_boundary, title='square')

p_boundary, _ = currents_along_circle(I=0, R=0.5, d_phi=step)
demo_stokes(p_boundary, title='circle', skip=250)

# -----------------------------
# Ejemplo 1: Campo simple rotacional
# -----------------------------
vx1, vy1, vz1 = -y, x, 0
compute_and_plot_curl(vx1, vy1, vz1, example_number=1)

# -----------------------------
# Ejemplo 2: Campo con variación cuadrática
# -----------------------------
vx2, vy2, vz2 = -y**2, x**2, 0
compute_and_plot_curl(vx2, vy2, vz2, example_number=2)
