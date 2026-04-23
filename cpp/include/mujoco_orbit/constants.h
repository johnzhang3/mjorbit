#ifndef MUJOCO_ORBIT_CONSTANTS_H_
#define MUJOCO_ORBIT_CONSTANTS_H_

// Units: km, s, kg, rad, T (standard astrodynamics).
// Ported from src/mujoco_orbit/constants.py.

namespace mujoco_orbit {

// Earth
inline constexpr double kGmEarth    = 398600.4418;    // km^3/s^2
inline constexpr double kREarth     = 6378.137;       // km
inline constexpr double kJ2Earth    = 1.08262668e-3;
inline constexpr double kOmegaEarth = 7.292115e-5;    // rad/s

// Sun
inline constexpr double kGmSun = 1.32712440018e11;    // km^3/s^2
inline constexpr double kAu    = 1.495978707e8;       // km

// Physics
inline constexpr double kCLight = 299792.458;         // km/s
inline constexpr double kPSun   = 4.56e-6;            // N/m^2
inline constexpr double kG0     = 9.80665e-3;         // km/s^2

// Magnetic
inline constexpr double kB0Earth = 3.12e-5;           // T

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_CONSTANTS_H_
