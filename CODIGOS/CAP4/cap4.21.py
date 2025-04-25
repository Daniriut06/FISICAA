import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Calcula el campo eléctrico en un conjunto de puntos 'p'
# generado por una distribución circular de carga de radio 1.
# Se aproxima como una suma de cargas puntuales distribuidas
# uniformemente en un círculo.
# -------------------------------------------------------------
def approx_circ(p, R=1, dq=0.05):
    phi_vals = np.arange(0, 2*np.pi, dq)  # Ángulos de cargas
    vf_total = np.zeros(p.shape)         # Inicializa campo total
    for phi in phi_vals:
        xq, yq = R * np.cos(phi), R * np.sin(phi)
        dx = p[0] - xq
        dy = p[1] - yq
        r = np.sqrt(dx**2 + dy**2)
        r = np.where(r == 0, 1e-10, r)  # Evita división por cero
        vf = np.vstack((dx / r**2, dy / r**2))  # Campo puntual
        vf_total += vf * dq  # Suma de campos (integral aproximada)
    return vf_total

# -------------------------------------------------------------
# Devuelve puntos a lo largo de un círculo de radio r
# -------------------------------------------------------------
def points_along_circle(r=1, step=0.2):
    phi = np.arange(0, 2*np.pi, step)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.vstack((x, y))

# -------------------------------------------------------------
# Dibuja flechas normales hacia fuera desde un contorno cerrado
# -------------------------------------------------------------
def plot_normals_enc(p, ax, delta=0.2):
    N = p.shape[1]
    for i in range(N):
        i2 = (i+1) % N
        dp = p[:,i2] - p[:,i]
        dp /= np.linalg.norm(dp)
        n = np.array([dp[1], -dp[0]])  # Vector normal hacia fuera
        xm = (p[0,i] + p[0,i2]) / 2
        ym = (p[1,i] + p[1,i2]) / 2
        ax.arrow(xm, ym, n[0]*delta, n[1]*delta,
                 head_width=0.1, color='black')

# -------------------------------------------------------------
# Código principal: visualización del campo de la carga circular
# -------------------------------------------------------------

scale = 0.8  # Escala de las flechas
step = 1     # Resolución de la cuadrícula
lim = 4.5    # Límite de los ejes

x, y = np.meshgrid(np.arange(-lim, lim+step, step),
                   np.arange(-lim, lim+step, step),
                   indexing='ij')
p = np.vstack((x.flatten(), y.flatten()))

vf = approx_circ(p)  # Calcula el campo en cada punto

fig = plt.figure(figsize=(3, 3))
plt.quiver(p[0], p[1], vf[0], vf[1],
           angles='xy', scale_units='xy', scale=scale)

# Dibuja el contorno circular para aplicar la ley de Gauss
p_circle = points_along_circle(r=2.5, step=0.25)
plt.plot(p_circle[0], p_circle[1], color='gray', linewidth=8, alpha=0.5)
plot_normals_enc(p_circle, plt.gca())

# Representa las cargas puntuales distribuidas en el círculo
for phi in np.arange(0, 2*np.pi, np.pi/7):
    x0, y0 = np.cos(phi), np.sin(phi)
    plt.scatter(x0, y0, marker='+', color='black')

# Ajustes del gráfico
plt.axis('equal')
plt.axis('square')
plt.xlim((-lim, lim))
plt.ylim((-lim, lim))
plt.xlabel('x')
plt.ylabel('y')
plt.xticks((-2, 0, 2))
plt.yticks((-2, 0, 2))
plt.text(+2.9, 0, 'D')
plt.text(+1.4, 0, 'R')
plt.savefig('fig_ch4_circle_gauss.pdf', bbox_inches='tight')
plt.show()
