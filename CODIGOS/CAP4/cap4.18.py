import numpy as np
import matplotlib.pyplot as plt

# Code Block 4.18: Funciones previas utilizadas en capítulos anteriores.

# Función para calcular los vectores normales a una frontera cerrada
def get_normals_enc(boundary, inside=(0, 0)):
    """
    Esta función calcula los vectores normales unitarios a lo largo de una frontera cerrada.
    
    :param boundary: array de puntos que definen la frontera cerrada.
    :param inside: punto de referencia dentro de la región cerrada.
    :return: matriz de vectores normales unitarios.
    """
    boundary_ext = np.hstack((boundary, boundary[:, :2]))  # Extiende la frontera para conectar el último punto con el primero.
    x, y = boundary_ext[0], boundary_ext[1]
    very_small_num = 10**(-10)  # Para evitar división por cero.
    
    # Calcula la pendiente de los segmentos de línea
    slope = (y[2:] - y[:-2]) / (x[2:] - x[:-2] + very_small_num)
    norm_vec_slope = -1 / (slope + very_small_num)  # Calcula la pendiente de los vectores normales
    u, v = 1, norm_vec_slope  # Componentes del vector normal
    mag = np.sqrt(u**2 + v**2)  # Magnitud del vector normal
    u, v = u / mag, v / mag  # Normaliza el vector
    n = np.vstack((u, v))  # Apila los componentes del vector normal
    
    # Determina la dirección correcta del vector normal
    x, y = x[1:-1] - inside[0], y[1:-1] - inside[1]
    dot_prod_sign = np.sign(x * n[0] + y * n[1])  # Producto punto para determinar la dirección
    n = n * dot_prod_sign  # Ajusta la dirección del vector normal
    return n

# Función para graficar los vectores normales
def plot_normals_enc(boundary, ax, inside=(0, 0), scale=3):
    """
    Esta función grafica los vectores normales a lo largo de una frontera cerrada.
    
    :param boundary: array de puntos que definen la frontera cerrada.
    :param ax: objeto de la figura para graficar.
    :param inside: punto de referencia dentro de la región cerrada.
    :param scale: factor de escala para los vectores normales.
    """
    n = get_normals_enc(boundary, inside=inside)
    boundary_ext = np.hstack((boundary, boundary[:, :2]))  # Extiende la frontera para conectar el último punto con el primero.
    x, y = boundary_ext[0], boundary_ext[1]
    color = '#CCCCCC'  # Color de los vectores normales
    ax.scatter(x, y, color='gray')  # Dibuja los puntos de la frontera
    ax.plot(x, y, color='gray')  # Dibuja la frontera cerrada
    ax.quiver(x[1:-1], y[1:-1], n[0], n[1], color=color, angles='xy', scale_units='xy', scale=scale)  # Dibuja los vectores normales
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.axis('square')

# Función para generar puntos sobre un círculo
def points_along_circle(r=1, step=np.pi/10):
    """
    Genera puntos distribuidos sobre un círculo de radio r.
    
    :param r: radio del círculo.
    :param step: paso angular entre los puntos generados.
    :return: array de puntos en coordenadas cartesianas.
    """
    phi = np.arange(-np.pi, np.pi, step)  # Ángulos en radianes
    p = np.vstack((r * np.cos(phi), r * np.sin(phi)))  # Coordenadas cartesianas
    return p

# Función para generar puntos sobre un cuadrado
def points_along_square(s=1, step=0.1):
    """
    Genera puntos distribuidos sobre los lados de un cuadrado de lado s.
    
    :param s: longitud del lado del cuadrado.
    :param step: paso entre los puntos generados.
    :return: array de puntos en coordenadas cartesianas.
    """
    d = s / 2
    one_side = np.arange(-d, d, step)  # Puntos a lo largo de un lado
    N = len(one_side)
    p_top = np.vstack((+one_side, np.zeros(N) + d))  # Lado superior
    p_rgt = np.vstack((np.zeros(N) + d, -one_side))  # Lado derecho
    p_bot = np.vstack((-one_side, np.zeros(N) - d))  # Lado inferior
    p_lft = np.vstack((np.zeros(N) - d, +one_side))  # Lado izquierdo
    p = np.hstack((p_top, p_rgt, p_bot, p_lft))  # Combina los puntos de los cuatro lados
    return p

# Función para calcular el flujo a través de una frontera cerrada
def get_flux_enc(boundary, vfield, inside=(0, 0)):
    """
    Calcula el flujo del campo eléctrico a través de una frontera cerrada.
    
    :param boundary: array de puntos que definen la frontera cerrada.
    :param vfield: campo eléctrico en cada punto de la frontera.
    :param inside: punto de referencia dentro de la región cerrada.
    :return: valor del flujo.
    """
    boundary_ext = np.hstack((boundary, boundary[:, :2]))  # Extiende la frontera
    x, y = boundary_ext[0], boundary_ext[1]
    dl_neighbor = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)  # Distancias entre puntos vecinos
    dl = (0.5) * (dl_neighbor[1:] + dl_neighbor[:-1])  # Promedio de longitudes
    n = get_normals_enc(boundary, inside=inside)  # Vectores normales
    xv, yv = vfield[0], vfield[1]  # Componentes del campo eléctrico
    dotprod = xv * n[0] + yv * n[1]  # Producto punto entre el campo y los normales
    flux = np.sum(dl * dotprod)  # Cálculo del flujo
    return flux

# Función para calcular el campo eléctrico radial en 2D (campo de una carga puntual)
def get_vfield_radial_2d(points, p_charge):
    """
    Calcula el campo eléctrico radial de una carga puntual en 2D.
    
    :param points: puntos en los que calcular el campo.
    :param p_charge: posición de la carga puntual.
    :return: campo eléctrico en cada punto.
    """
    x_charge, y_charge = p_charge
    dx = points[0] - x_charge
    dy = points[1] - y_charge
    r = np.sqrt(dx**2 + dy**2)  # Distancia radial a la carga
    r[r == 0] = 1e-10  # Evita la singularidad en el centro de la carga
    Ex = (dx / r**3)  # Componente x del campo
    Ey = (dy / r**3)  # Componente y del campo
    return np.vstack((Ex, Ey))

# Verificación de la ley de Gauss numéricamente
def verify_gauss(q, p_charges, boundary_type='circle'):
    """
    Verifica la ley de Gauss numéricamente para cargas puntuales.
    
    :param q: signos de las cargas (1 o -1).
    :param p_charges: ubicaciones de las cargas puntuales.
    :param boundary_type: tipo de frontera (círculo o cuadrado).
    """
    # Genera los puntos sobre la frontera según el tipo
    if boundary_type == 'circle':
        p = points_along_circle(r=1, step=0.001)  # Puntos sobre un círculo
        p_plot = points_along_circle(r=1, step=0.3)  # Puntos para graficar
    else:
        p = points_along_square(s=2, step=0.001)  # Puntos sobre un cuadrado
        p_plot = points_along_square(s=2, step=0.3)  # Puntos para graficar
    
    total_vf = np.zeros(p.shape)  # Campo eléctrico total
    total_vf_plot = np.zeros(p_plot.shape)  # Campo eléctrico total para graficar
    
    # Calcular el campo eléctrico debido a cada carga puntual
    for i, p_charge in enumerate(p_charges):
        vf, _ = get_vfield_radial_2d(p, p_charge)  # Campo eléctrico en p
        total_vf = total_vf + vf * q[i]  # Suma de los campos ponderados por la carga
        vf, _ = get_vfield_radial_2d(p_plot, p_charge)  # Campo eléctrico en p_plot
        total_vf_plot = total_vf_plot + vf * q[i]  # Suma de los campos ponderados por la carga
    
    # Calcular el flujo
    flux = get_flux_enc(p, total_vf)
    
    # Graficar
    plt.figure(figsize=(3, 3))
    for i in range(len(q)):
        marker = '+' if q[i] > 0 else '_'
        plt.scatter(p_charges[i, 0], p_charges[i, 1], marker=marker, s=100, color='black')
    
    plot_normals_enc(p_plot, plt.gca(), inside=(0, 0))  # Graficar los vectores normales
    plt.quiver(p_plot[0], p_plot[1], total_vf_plot[0], total_vf_plot[1], color='black', scale=2)
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.xticks((-1, 0, 1))
    plt.yticks((-1, 0, 1))
    plt.title(f'Flux = {flux:.3f}')
    
    # Mostrar la figura
    plt.show()

# Ejemplos de uso
verify_gauss([1], np.array([[0, 0]]))
verify_gauss([1], np.array([[0.4, 0.4]]))
verify_gauss([1], np.array([[1.1, 1.1]]))
verify_gauss([1], np.array([[0, 0]]), boundary_type='square')
verify_gauss([1], np.array([[0.4, 0.4]]), boundary_type='square')
verify_gauss([1], np.array([[0.7, 1.4]]), boundary_type='square')
verify_gauss([1, -1], np.array([[-0.4, -0.4], [0.3, 0.3]]))
verify_gauss([1, -1, 1], np.array([[0, 0], [0.3, 0.3], [-0.2, 0.1]]))
verify_gauss([1, -1, -1, 1], np.array([[0, 0], [0.3, 0.3], [-0.4, 0.1], [-0.1, -0.4]]))
verify_gauss([1, 1, -1, 1], np.array([[0, 0], [-0.2, 0.2], [-0.4, 0.1], [1.4, 1.4]]))
verify_gauss([1, -1, -1, 1], np.array([[0, 0], [0.3, 0.3], [-0.4, 0.1], [-0.1, -0.4]]), boundary_type='square')
verify_gauss([1, 1, -1, 1], np.array([[0, 0], [-0.2, 0.2], [-0.4, 0.1], [1.4, 1.4]]), boundary_type='square')
