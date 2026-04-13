import numpy as np
from numba import njit

SUN_MU_KM = 132712440018.0  # km^3 / s^2

@njit
def mag(vec):
    """Fast vector magnitude."""
    return np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

@njit
def cross(a, b):
    """Fast 3D cross product."""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

@njit
def solve_transfer(r1, r2, tof_seconds, mu=SUN_MU_KM, tol=1e-6, max_iter=100):
    """
    Custom Heliocentric Linker (p-iteration method).
    Finds the required departure and arrival velocities to connect r1 and r2 in tof_seconds.
    """
    r1_mag = mag(r1)
    r2_mag = mag(r2)
    
    # Calculate the change in true anomaly (angle between r1 and r2)
    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    # Clip to prevent floating point errors for acos
    cos_dnu = max(-1.0, min(1.0, cos_dnu))
    dnu = np.arccos(cos_dnu)
    
    # Determine if we are going the "short way" or "long way" around the Sun
    if cross(r1, r2)[2] < 0:
        dnu = 2 * np.pi - dnu
        
    sin_dnu = np.sin(dnu)
    
    # Chord length between planets
    k = r1_mag * r2_mag * (1 - cos_dnu)
    l = r1_mag + r2_mag
    m = r1_mag * r2_mag * (1 + cos_dnu)
    
    # Initial guess for p (semi-latus rectum)
    p_min = k / (l + np.sqrt(2 * m))
    p_max = k / (l - np.sqrt(2 * m)) if dnu < np.pi else 1e10
    
    p = (p_min + r1_mag) / 2.0  # Safe initial guess
    
    # Newton-Raphson iteration to find the right orbit width (p) that matches our Time of Flight
    for _ in range(max_iter):
        a = m * k * p / ((2 * m - l**2) * p**2 + 2 * k * l * p - k**2)
        
        # Calculate f and g functions (Lagrange coefficients)
        f = 1 - (r2_mag / p) * (1 - cos_dnu)
        g = (r1_mag * r2_mag * sin_dnu) / np.sqrt(mu * p)
        
        # Calculate time of flight for this p
        f_dot = np.sqrt(mu / p) * np.tan(dnu / 2) * ((1 - cos_dnu) / p - 1 / r1_mag - 1 / r2_mag)
        g_dot = 1 - (r1_mag / p) * (1 - cos_dnu)
        
        # Determine eccentric anomaly change (simplified for elliptical orbits here)
        cos_dE = 1 - (r1_mag * r2_mag / a) * (1 - cos_dnu)
        cos_dE = max(-1.0, min(1.0, cos_dE))
        dE = np.arccos(cos_dE)
        if dnu > np.pi:
            dE = 2 * np.pi - dE
            
        t_guess = g + np.sqrt(a**3 / mu) * (dE - np.sin(dE))
        
        # If our guess is close enough to the target Time of Flight, we win!
        if abs(t_guess - tof_seconds) < tol:
            break
            
        # Numba-friendly derivative of time with respect to p for Newton step
        # (Using a simple numerical gradient for stability)
        dp = 1e-5
        p_plus = p + dp
        a_plus = m * k * p_plus / ((2 * m - l**2) * p_plus**2 + 2 * k * l * p_plus - k**2)
        cos_dE_plus = 1 - (r1_mag * r2_mag / a_plus) * (1 - cos_dnu)
        cos_dE_plus = max(-1.0, min(1.0, cos_dE_plus))
        dE_plus = np.arccos(cos_dE_plus)
        if dnu > np.pi: dE_plus = 2 * np.pi - dE_plus
        g_plus = (r1_mag * r2_mag * sin_dnu) / np.sqrt(mu * p_plus)
        t_plus = g_plus + np.sqrt(a_plus**3 / mu) * (dE_plus - np.sin(dE_plus))
        
        dt_dp = (t_plus - t_guess) / dp
        p = p - (t_guess - tof_seconds) / dt_dp
        
        # Keep p strictly positive and above minimum
        p = max(p_min * 1.01, p)

    # Final velocities required to make the transfer
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    
    return v1, v2