#ifndef MJORBIT_CONSTANTS_H_
#define MJORBIT_CONSTANTS_H_

// Units: km, s, kg, rad, T (standard astrodynamics).
// Ported from src/mjorbit/constants.py.

namespace mjorbit {

// Earth
inline constexpr double kGmEarth    = 398600.4418;    // km^3/s^2
inline constexpr double kREarth     = 6378.137;       // km
inline constexpr double kJ2Earth    = 1.08262668e-3;
inline constexpr double kOmegaEarth = 7.292115e-5;    // rad/s
// Earth Rotation Angle at J2000.0 (rad): 2*pi * 0.7790572732640 (IERS 2010).
inline constexpr double kEraJ2000 = 4.894961212823756;

// Sun
inline constexpr double kGmSun = 1.32712440018e11;    // km^3/s^2
inline constexpr double kAu    = 1.495978707e8;       // km

// Physics
inline constexpr double kCLight = 299792.458;         // km/s
inline constexpr double kPSun   = 4.56e-6;            // N/m^2
inline constexpr double kG0     = 9.80665e-3;         // km/s^2

// Magnetic
inline constexpr double kB0Earth = 3.12e-5;           // T

}  // namespace mjorbit

#endif  // MJORBIT_CONSTANTS_H_
