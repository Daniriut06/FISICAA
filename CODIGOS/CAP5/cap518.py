"""
Code Blocks 5.17-5.18 - Non-uniform current density in a cylindrical wire

This implementation models a cylindrical wire with current density J ~ r² dependence,
providing three different methods for discretizing the current distribution:
1. Polar coordinate sampling
2. Cartesian coordinate sampling
3. Random sampling

The code also includes visualization functions to show the sampling points.
"""

import numpy as np
import matplotlib.pyplot as plt

def r2_curr_from_p_curr(I, R, L, p_curr, dV):
    """
    Calculate current vector for each point based on r² dependence
    
    Parameters:
        I: Total current [A]
        R: Wire radius [m]
        L: Wire length [m]
        p_curr: Array of current point positions (3 x N)
        dV: Volume element for each point [m³]
    
    Returns:
        Current vector array (3 x N) where only z-component is non-zero
    """
    r = np.sqrt(p_curr[0]**2 + p_curr[1]**2)  # Radial distance
    alpha = (2*I)/(np.pi*(R**4))  # Proportionality constant from previous calculation
    N = len(r)
    curr = np.zeros((3, N))
    curr[2] = alpha * (r**2) * dV  # J_z = α·r²
    return curr

def curr_density_r2_polar(I, R, L, dR, dL):
    """
    Generate current points using polar coordinate sampling
    
    Parameters:
        I: Total current [A]
        R: Wire radius [m]
        L: Wire length [m]
        dR: Radial step size [m]
        dL: Longitudinal step size [m]
    
    Returns:
        p_curr: Position array (3 x N)
        curr: Current vector array (3 x N)
    """
    dphi = np.pi/32  # Angular step size
    r, phi, z = np.meshgrid(
        np.arange(0, R+dR, dR),
        np.arange(0, 2*np.pi, dphi),
        np.arange(-L/2, L/2+dL, dL),
        indexing='ij'
    )
    
    # Convert to Cartesian coordinates
    x, y = r*np.cos(phi), r*np.sin(phi)
    p_curr = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    dV = r.flatten() * dR * dphi * dL  # Volume element in polar coords
    curr = r2_curr_from_p_curr(I, R, L, p_curr, dV)
    return p_curr, curr

def curr_density_r2_cartesian(I, R, L, dR, dL):
    """
    Generate current points using Cartesian coordinate sampling
    
    Parameters: Same as polar version
    Returns: Same as polar version
    """
    v = np.arange(-R, R+dR, dR)
    x, y, z = np.meshgrid(
        v, v, 
        np.arange(-L/2, L/2+dL, dL),
        indexing='ij'
    )
    
    # Filter points outside the cylinder
    p_tmp = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    r_tmp = np.sqrt(p_tmp[0]**2 + p_tmp[1]**2)
    mask = r_tmp <= R
    p_curr = p_tmp[:, mask]
    dV = (dR**2) * dL  # Volume element in Cartesian coords
    curr = r2_curr_from_p_curr(I, R, L, p_curr, dV)
    return p_curr, curr

def curr_density_r2_random(I, R, L, N=10000):
    """
    Generate current points using random sampling
    
    Parameters:
        N: Number of random points
        Others same as polar version
    Returns: Same as polar version
    """
    # Use sqrt for uniform radial distribution
    r = np.sqrt(np.random.rand(N)) * R
    phi = np.random.rand(N) * 2*np.pi
    z = (np.random.rand(N)*2 - 1) * L/2  # Uniform in [-L/2, L/2]
    
    # Convert to Cartesian
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    p_curr = np.vstack((x, y, z))
    dV = (np.pi * R**2 * L) / N  # Total volume divided by N points
    curr = r2_curr_from_p_curr(I, R, L, p_curr, dV)
    return p_curr, curr

def show_r2_density(I, R, L, p_curr, curr, title=''):
    """
    Visualize the current source points in 2D projections
    
    Parameters:
        I, R, L: Same as before
        p_curr: Position array
        curr: Current array
        title: Figure title suffix
    """
    r = np.sqrt(p_curr[0]**2 + p_curr[1]**2)
    curr_r = curr[2]
    axis_lim = np.array([-R, R]) * 1.1
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4), 
                                gridspec_kw={'width_ratios': [1, 2]})
    
    # XY projection
    ax0.scatter(p_curr[0], p_curr[1], s=1, color='black')
    ax0.set(title='XY Plane', xlabel='x [m]', ylabel='y [m]',
           xlim=axis_lim, ylim=axis_lim, aspect='equal')
    ax0.set_xticks([-R, 0, R])
    ax0.set_yticks([-R, 0, R])
    
    # ZY projection
    ax1.scatter(p_curr[2], p_curr[1], s=1, color='black')
    ax1.set(title='ZY Plane', xlabel='z [m]', ylabel='y [m]',
           xlim=[-L/2*1.1, L/2*1.1], ylim=axis_lim)
    ax1.set_xticks([-L/2, 0, L/2])
    ax1.set_yticks([-R, 0, R])
    
    fig.suptitle(f'Current Density Sampling (N = {len(r):,}) - {title.capitalize()} Coordinates')
    plt.tight_layout()
    plt.savefig(f'current_density_sampling_{title}.png', dpi=150, bbox_inches='tight')
    plt.show()

# Parameters
I = 1.0  # Total current [A]
R = 0.5  # Wire radius [m]
L = 10.0  # Wire length [m]
dR, dL = R/10, L/30  # Spatial steps

# Generate and visualize different sampling methods
print("=== Polar Coordinate Sampling ===")
p_curr_polar, curr_polar = curr_density_r2_polar(I, R, L, dR, dL)
show_r2_density(I, R, L, p_curr_polar, curr_polar, 'polar')

print("\n=== Cartesian Coordinate Sampling ===")
p_curr_cart, curr_cart = curr_density_r2_cartesian(I, R, L, dR, dL)
show_r2_density(I, R, L, p_curr_cart, curr_cart, 'cartesian')

print("\n=== Random Sampling ===")
p_curr_rand, curr_rand = curr_density_r2_random(I, R, L, N=5000)
show_r2_density(I, R, L, p_curr_rand, curr_rand, 'random')