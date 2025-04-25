import numpy as np
import matplotlib.pyplot as plt

def get_vfield_radial_2d(p, source):
    """Calculate radial vector field from a point source"""
    dx = p[0] - source[0]
    dy = p[1] - source[1]
    r = np.sqrt(dx**2 + dy**2)
    vx = dx / (r**3)
    vy = dy / (r**3)
    return np.vstack((vx, vy)), p

def points_along_square(s=1, step=0.1):
    """Generate points along the perimeter of a square"""
    side = np.arange(-s/2, s/2+step, step)
    top = np.vstack((side, np.full_like(side, s/2)))
    right = np.vstack((np.full_like(side, s/2), side))
    bottom = np.vstack((side[::-1], np.full_like(side, -s/2)))
    left = np.vstack((np.full_like(side, -s/2), side[::-1]))
    return np.hstack((top, right, bottom, left))

def plot_normals_enc(points, ax):
    """Plot normal vectors for points on a square"""
    n = points.shape[1]
    for i in range(n):
        x, y = points[:, i]
        if x == max(points[0]):  # right side
            ax.quiver(x, y, 1, 0, color='red', width=0.005)
        elif x == min(points[0]):  # left side
            ax.quiver(x, y, -1, 0, color='red', width=0.005)
        elif y == max(points[1]):  # top side
            ax.quiver(x, y, 0, 1, color='red', width=0.005)
        elif y == min(points[1]):  # bottom side
            ax.quiver(x, y, 0, -1, color='red', width=0.005)

# Main visualization
fig = plt.figure(figsize=(3,3))

# Infinite line parameters
D = 0.5
step = 0.1

# Create grid of points
x, y = np.meshgrid(np.arange(-1, 1+step, step),
                   np.arange(-1, 1+step, step),
                   indexing='ij')
p = np.vstack((x.flatten(), y.flatten()))

# Avoid choosing positions along the y-axis
idx = np.where(np.abs(p[0]) > 0.01)
p = np.vstack((p[0,idx], p[1,idx]))

# Calculate vector field from infinite line
dy = 0.1
y_max = 10
y_range = np.arange(-y_max, y_max, dy)  # Practically infinite
vf_total = np.zeros(p.shape)

for ys in y_range:
    vf, p = get_vfield_radial_2d(p, (0, ys))
    vf_total = vf_total + (vf) * dy

# Plot the square
plt.plot([-D, D, D, -D, -D], [D, D, -D, -D, D], 
         color='gray', linewidth=8, alpha=0.5)

# Plot normals on square perimeter
p_square = points_along_square(s=D*2, step=0.2)
plot_normals_enc(p_square, plt.gca())

# Plot the line sources
plt.scatter(np.zeros(y_range.shape), y_range,
            marker='+', color='black')

# Plot the vector field
plt.quiver(p[0], p[1], vf_total[0], vf_total[1], 
           color='black', scale=20, width=0.003)

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()