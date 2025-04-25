"""
Este script verifica que el campo eléctrico es cero dentro de una shell circular y que disminuye como 1/r fuera de la shell.

Se calcula el campo eléctrico en varios puntos a lo largo del eje x y se visualiza la magnitud del campo eléctrico. También se realiza un ajuste log-log para verificar la relación de caída del campo eléctrico.

El código realiza las siguientes tareas:

1. **Definición de puntos**: Se definen puntos a lo largo del eje x para calcular el campo eléctrico.

2. **Cálculo del campo eléctrico**: Se calcula el campo eléctrico en los puntos definidos utilizando la función `approx_circ`.

3. **Visualización**: Se grafican los resultados, mostrando la magnitud del campo eléctrico y la relación de caída.

4. **Ajuste log-log**: Se realiza un ajuste lineal en una escala logarítmica para verificar la relación de caída del campo eléctrico.
"""

"""
Este script verifica que el campo eléctrico es cero dentro de una shell circular y que disminuye como 1/r fuera de la shell.

Se calcula el campo eléctrico en varios puntos a lo largo del eje x y se visualiza la magnitud del campo eléctrico. También se realiza un ajuste log-log para verificar la relación de caída del campo eléctrico.

El código realiza las siguientes tareas:

1. **Definición de puntos**: Se definen puntos a lo largo del eje x para calcular el campo eléctrico.

2. **Cálculo del campo eléctrico**: Se calcula el campo eléctrico en los puntos definidos utilizando la función `approx_circ`.

3. **Visualización**: Se grafican los resultados, mostrando la magnitud del campo eléctrico y la relación de caída.

4. **Ajuste log-log**: Se realiza un ajuste lineal en una escala logarítmica para verificar la relación de caída del campo eléctrico.
"""

import numpy as np
import matplotlib.pyplot as plt

def get_vfield_radial_2d(p, p_charge):
    x, y = p[0] - p_charge[0], p[1] - p_charge[1]
    r = np.sqrt(x**2 + y**2)
    valid_idx = np.where(r > np.max(r) * 0.01)
    p, r = p[:, valid_idx].squeeze(), r[valid_idx].squeeze()
    vf = np.zeros(p.shape)
    vf[0] = (p[0] - p_charge[0]) / r**2
    vf[1] = (p[1] - p_charge[1]) / r**2
    vf = vf / (2 * np.pi)
    return vf, p

def approx_circ(p, r=1, d_phi=2 * np.pi / 200):
    phi_range = np.arange(0, 2 * np.pi, d_phi)
    vf = np.zeros(p.shape)
    for phi in phi_range:
        x0, y0 = r * np.cos(phi), r * np.sin(phi)
        vf0, _ = get_vfield_radial_2d(p, (x0, y0))
        vf += vf0 * r * d_phi
    return vf

# Definición de puntos a lo largo del eje x
px = np.array([0, 0.2, 0.4, 0.8, 1.2, 1.4, 1.6, 1.8, 2, 4, 8, 16])
px = np.hstack((-px, px))
py = np.zeros(len(px))
p = np.vstack((px, py))

# Cálculo del campo eléctrico
d_phi = 2 * np.pi / 200
vf = approx_circ(p, d_phi=d_phi)
vf_mag = np.sqrt(np.sum(vf**2, axis=0))

# Visualización del campo eléctrico
fig = plt.figure(figsize=(3, 3))
plt.scatter(px, vf_mag, color='black', zorder=2)
q_enc = 1
D_range = np.arange(1.2, 16, 0.01)
plt.plot(+D_range, q_enc / D_range, color='black')
plt.plot(-D_range, q_enc / D_range, color='black')

# Área sombreada donde E = 0
xbox = np.array([-1, -1, 1, 1])
ybox = np.array([0, q_enc / D_range[0], q_enc / D_range[0], 0])
plt.fill(xbox, ybox, color='#CCCCCC', zorder=1)

# Ajustes del gráfico
plt.xlabel('D')
plt.ylabel('|E|')
plt.xticks(np.arange(-16, 18, 8))
plt.title('E field from a circular charge distribution')
plt.savefig('fig_ch4_circle_E.pdf', bbox_inches='tight')
plt.show()

# Ajuste log-log
logx, logy = np.log(px[px > 1]), np.log(vf_mag[px > 1])
fig = plt.figure(figsize=(3, 3))
plt.scatter(logx, logy, color='black')
pfit = np.polyfit(logx, logy, 1)
plt.plot(logx, logx * pfit[0] + pfit[1], color='black')
plt.xlim((0, 3))
plt.xticks((0, 1, 2, 3))
plt.xlabel('log(D)')
plt.ylabel('log(|E|)')
plt.title('log-log plot (outside)')
plt.legend(('Slope = %4.3f' % pfit[0], 'log(|E|)'))
plt.savefig('fig_ch4_circle_logE.pdf', bbox_inches='tight')
plt.show()