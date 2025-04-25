# Visualización de la circulación de un campo vectorial en distintos casos.
# Este script grafica el campo vectorial, una "veleta" que indica el sentido de la circulación,
# y los vectores tangentes sobre una trayectoria circular cerrada. Se evalúan tres casos: 
# campo más intenso arriba, uniforme y más intenso abajo.

import numpy as np
import matplotlib.pyplot as plt

# Función para definir un campo vectorial artificial con tres configuraciones distintas.
def get_vfield_example(p, case=1):
    strength = 50  # intensidad base
    vfield = np.zeros(p.shape)
    if case == 0:  # viento más fuerte arriba
        vfield[0] = strength * (1 + p[1])
    elif case == 1:  # viento uniforme
        vfield[0] = strength * 2
    elif case == 2:  # viento más fuerte abajo
        vfield[0] = strength * (1 - p[1])
    return vfield

# Cálculo del integral de línea (circulación) sobre el campo vectorial y una trayectoria cerrada.
def line_integral_vector_field(vfield, path):
    dr = np.diff(path)
    f_avg = 0.5 * (vfield[:, :-1] + vfield[:, 1:])
    integrand = np.sum(f_avg * dr, axis=0)
    return np.sum(integrand)

# Crear trayectoria circular cerrada centrada en el origen.
def currents_along_circle(I=0, R=0.5, d_phi=np.pi/128):
    phi = np.arange(0, 2 * np.pi, d_phi)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    path = np.vstack((x, y))
    return path, phi

# Función para graficar el campo completo.
def plot_full_field(ax, p, vfield, scale=250):
    ax.quiver(p[0], p[1], vfield[0], vfield[1],
              angles='xy', scale_units='xy', scale=scale)

# Función para graficar campo sobre la trayectoria.
def plot_around_path(ax, path, vfield, scale=250):
    s = 16
    ax.plot(path[0], path[1], color='gray', linewidth=1)
    ax.quiver(path[0][::s], path[1][::s],
              vfield[0][::s], vfield[1][::s],
              angles='xy', scale_units='xy', scale=scale)

# Función para graficar la "veleta" que indica circulación

def plot_windmill(ax, rotate_mag=0):
    R = 1.0
    c = ('#EEEEEE', '#CCCCCC', '#AAAAAA', '#000000')
    phi_range = np.array([0, 1, 2, 3]) * np.pi / 20 * rotate_mag
    for i, phi in enumerate(phi_range):
        ax.plot([-R * np.cos(-phi), +R * np.cos(-phi)],
                [-R * np.sin(-phi), +R * np.sin(-phi)],
                color=c[i], linewidth=6)
        ax.plot([-R * np.cos(-phi + np.pi/2), +R * np.cos(-phi + np.pi/2)],
                [-R * np.sin(-phi + np.pi/2), +R * np.sin(-phi + np.pi/2)],
                color=c[i], linewidth=6)

# Configuración estética del gráfico

def tidy_axis(ax):
    lim = 1.25
    ax.axis('square')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))

# Crear trayectoria circular y puntos del campo
path, _ = currents_along_circle(I=0, R=0.5, d_phi=np.pi/128)
x, y = np.meshgrid(np.arange(-1, 1, 0.5), np.arange(-1, 1.5, 0.5), indexing='ij')
p = np.vstack((x.flatten(), y.flatten()))

scale = 250
fig, axs = plt.subplots(3, 3, figsize=(5, 5), sharey=True, sharex=True)

for i in range(3):
    full_vfield = get_vfield_example(p, case=i)
    vfield = get_vfield_example(path, case=i)
    circulation = line_integral_vector_field(vfield, path)

    plot_full_field(axs[0, i], p, full_vfield, scale=scale)
    tidy_axis(axs[0, i])

    plot_windmill(axs[1, i], rotate_mag=circulation)
    tidy_axis(axs[1, i])

    plot_around_path(axs[2, i], path, vfield, scale=scale)
    tidy_axis(axs[2, i])

    axs[2, i].set_title("%3.2f" % circulation)

axs[0, 0].set_title('Top-heavy')
axs[0, 1].set_title('Uniform')
axs[0, 2].set_title('Bottom-heavy')
axs[1, 0].set_title('Clockwise')
axs[1, 1].set_title('No Rotation')
axs[1, 2].set_title('Counterclockwise')
axs[0, 0].set_ylabel('Full Field')
axs[1, 0].set_ylabel('Windmill')
axs[2, 0].set_ylabel('Path')

plt.tight_layout()
plt.savefig('fig_ch5_wind_circulation.pdf', bbox_inches='tight')
plt.show()
