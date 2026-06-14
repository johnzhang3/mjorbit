"""Physical constants for mjorbit.

Units: km, s, kg, rad, T (standard astrodynamics).
Ported from OrbitX constants.py.
"""

# Earth
GM_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137  # km
J2_EARTH = 1.08262668e-3
OMEGA_EARTH = 7.292115e-5  # rad/s

# Sun
GM_SUN = 1.32712440018e11  # km^3/s^2
AU = 1.495978707e8  # km

# Physics
C_LIGHT = 299792.458  # km/s
P_SUN = 4.56e-6  # N/m^2  solar radiation pressure at 1 AU
G0 = 9.80665e-3  # km/s^2  standard gravity

# Magnetic
B0_EARTH = 3.12e-5  # T  Earth's magnetic dipole moment magnitude (equatorial surface)
