import numpy as np
import matplotlib.pyplot as plt

# ================== FUNCTIONS FOR POINT GENERATION ==================

def points_cartesian_xy(xmin=-1, xmax=1, ymin=-1, ymax=1, delta=0.1):
    """
    Generate observation points in the xy-plane (z=0)
    Parameters:
        xmin, xmax: x-range
        ymin, ymax: y-range
        delta: spacing between points
    Returns:
        3xN array of points
    """
    x = np.arange(xmin, xmax+delta, delta)
    y = np.arange(ymin, ymax+delta, delta)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    zv = np.zeros_like(xv)
    return np.vstack((xv.flatten(), yv.flatten(), zv.flatten()))

def points_cartesian_xz(xmin=-1, xmax=1, zmin=-1, zmax=1, delta=0.1):
    """
    Generate observation points in the xz-plane (y=0)
    Parameters:
        xmin, xmax: x-range
        zmin, zmax: z-range
        delta: spacing between points
    Returns:
        3xN array of points
    """
    x = np.arange(xmin, xmax+delta, delta)
    z = np.arange(zmin, zmax+delta, delta)
    xv, zv = np.meshgrid(x, z, indexing='ij')
    yv = np.zeros_like(xv)
    return np.vstack((xv.flatten(), yv.flatten(), zv.flatten()))

# ================== CURRENT ELEMENT GENERATORS ==================

def currents_along_line(I=1, L=1, dL=0.1):
    """
    Create current elements along a straight line (z-axis)
    """
    z = np.arange(-L/2, L/2 + dL, dL)
    p_curr = np.vstack((np.zeros_like(z), np.zeros_like(z), z))
    curr = np.vstack((np.zeros_like(z), np.zeros_like(z), I*dL*np.ones_like(z)))
    return p_curr, curr

def currents_along_square(I=1, L=1, dL=0.1):
    """
    Create current elements along a square loop in xy-plane
    """
    # Right side (x = L/2, y varies)
    y_right = np.arange(-L/2, L/2, dL)
    p_right = np.vstack((L/2*np.ones_like(y_right), y_right, np.zeros_like(y_right)))
    curr_right = np.vstack((np.zeros_like(y_right), I*dL*np.ones_like(y_right), np.zeros_like(y_right)))
    
    # Top side (y = L/2, x varies)
    x_top = np.arange(L/2, -L/2, -dL)
    p_top = np.vstack((x_top, L/2*np.ones_like(x_top), np.zeros_like(x_top)))
    curr_top = np.vstack((-I*dL*np.ones_like(x_top), np.zeros_like(x_top), np.zeros_like(x_top)))
    
    # Left side (x = -L/2, y varies)
    y_left = np.arange(L/2, -L/2, -dL)
    p_left = np.vstack((-L/2*np.ones_like(y_left), y_left, np.zeros_like(y_left)))
    curr_left = np.vstack((np.zeros_like(y_left), -I*dL*np.ones_like(y_left), np.zeros_like(y_left)))
    
    # Bottom side (y = -L/2, x varies)
    x_bottom = np.arange(-L/2, L/2, dL)
    p_bottom = np.vstack((x_bottom, -L/2*np.ones_like(x_bottom), np.zeros_like(x_bottom)))
    curr_bottom = np.vstack((I*dL*np.ones_like(x_bottom), np.zeros_like(x_bottom), np.zeros_like(x_bottom)))
    
    # Combine all sides
    p_curr = np.hstack((p_right, p_top, p_left, p_bottom))
    curr = np.hstack((curr_right, curr_top, curr_left, curr_bottom))
    
    return p_curr, curr

def currents_along_circle(I=1, R=1, dphi=np.pi/20):
    """
    Create current elements along a circular ring in xy-plane
    """
    phi = np.arange(0, 2*np.pi, dphi)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    p_curr = np.vstack((x, y, np.zeros_like(x)))
    curr = np.vstack((-np.sin(phi), np.cos(phi), np.zeros_like(x))) * I * R * dphi
    return p_curr, curr

# ================== MAGNETIC FIELD CALCULATION ==================

def get_magnetic_field(p, p_curr, curr):
    """
    Calculate magnetic field using Biot-Savart law
    Parameters:
        p: observation points (3xN)
        p_curr: current element positions (3xM)
        curr: current vectors (3xM)
    Returns:
        Magnetic field at observation points (3xN)
    """
    mu0 = 4*np.pi*1e-7  # Permeability of free space
    B = np.zeros_like(p)
    
    for i in range(p.shape[1]):
        for j in range(p_curr.shape[1]):
            r = p[:,i] - p_curr[:,j]
            r_norm = np.linalg.norm(r)
            
            if r_norm < 1e-10:  # Avoid division by zero
                continue
                
            dB = (mu0/(4*np.pi)) * np.cross(curr[:,j], r) / (r_norm**3)
            B[:,i] += dB
            
    return B

# ================== VISUALIZATION ==================

def plot_magnetic_field_2d(p, p_curr, curr, B, view='xy', title=''):
    """
    Plot 2D magnetic field in either xy or xz plane
    Parameters:
        p: observation points (3xN)
        p_curr: current element positions (3xM)
        curr: current vectors (3xM)
        B: magnetic field vectors (3xN)
        view: 'xy' or 'xz' plane
        title: plot title
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    
    scale = 7*(10**(-6))  # Scaling factor for quiver arrows
    lw = 0.5  # Line width for quiver arrows
    
    if view == 'xy':
        # Plot in xy-plane (z=0)
        idx = np.where(np.sqrt(p[0]**2 + p[1]**2) >= 0.15)  # Exclude points near origin
        ax.quiver(p[0,idx], p[1,idx], B[0,idx], B[1,idx],
                 angles='xy', scale_units='xy', scale=scale,
                 linewidth=lw, zorder=0)
        ax.set_ylabel('y')
        
    elif view == 'xz':
        # Plot in xz-plane (y=0)
        idx = np.where(np.abs(p[2]) >= 0.02)  # Exclude points near xy-plane
        ax.quiver(p[0,idx], p[2,idx], B[0,idx], B[2,idx],
                 angles='xy', scale_units='xy', scale=scale,
                 linewidth=lw, zorder=0)
        ax.set_ylabel('z')
    
    ax.set_title(title)
    ax.axis('square')
    ax.set_xlabel('x')
    ax.set_xticks((-1, 0, 1))
    ax.set_yticks((-1, 0, 1))
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    
    return fig, ax

# ================== MAIN EXECUTION ==================

if __name__ == "__main__":
    # 1. Straight wire example
    print("Calculating field for straight wire...")
    I, L, dL = 1, 5, 0.1
    p_curr, curr = currents_along_line(I, L, dL)
    p = points_cartesian_xy()
    B = get_magnetic_field(p, p_curr, curr)
    fig, ax = plot_magnetic_field_2d(p, p_curr, curr, B, view='xy', title='Straight Wire (xy-plane)')
    plt.scatter([0], [0], s=400, color='gray')  # Mark wire position
    plt.tight_layout()
    plt.savefig('fig_ch5_B_line_2d.pdf', bbox_inches='tight')
    plt.show()

    # 2. Square loop example
    print("Calculating field for square loop...")
    I, L, dL = 1, 1, 0.01
    p_curr, curr = currents_along_square(I, L, dL)
    p = points_cartesian_xz(zmin=-0.9, zmax=0.9, delta=0.1)
    B = get_magnetic_field(p, p_curr, curr)
    fig, ax = plot_magnetic_field_2d(p, p_curr, curr, B, view='xz', title='Square Loop (xz-plane)')
    plt.plot([-L/2, L/2], [0, 0], linewidth=15, color='gray', zorder=1)
    plt.tight_layout()
    plt.savefig('fig_ch5_B_square_2d.pdf', bbox_inches='tight')
    plt.show()

    # 3. Ring example
    print("Calculating field for ring...")
    I, R = 1, 0.6
    p_curr, curr = currents_along_circle(I, R, np.pi/100)
    p = points_cartesian_xz(zmin=-0.9, zmax=0.9, delta=0.1)
    B = get_magnetic_field(p, p_curr, curr)
    fig, ax = plot_magnetic_field_2d(p, p_curr, curr, B, view='xz', title='Circular Ring (xz-plane)')
    plt.plot([-R, R], [0, 0], linewidth=15, color='gray', zorder=1)
    plt.tight_layout()
    plt.savefig('fig_ch5_B_ring_2d.pdf', bbox_inches='tight')
    plt.show()