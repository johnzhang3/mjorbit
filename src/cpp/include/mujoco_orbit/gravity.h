#ifndef MUJOCO_ORBIT_GRAVITY_H_
#define MUJOCO_ORBIT_GRAVITY_H_

#include "mujoco_orbit/constants.h"
#include "mujoco_orbit/spec.h"

namespace mujoco_orbit {

void point_mass_accel(const double r_eci[3], double out_a[3], double gm = kGmEarth);

void j2_accel(
    const double r_eci[3],
    double out_a[3],
    double gm = kGmEarth,
    double j2 = kJ2Earth,
    double r_eq = kREarth);

void total_accel(
    const double r_eci[3],
    double out_a[3],
    bool use_j2 = true,
    double gm = kGmEarth,
    double j2 = kJ2Earth,
    double r_eq = kREarth);

void total_accel(
    const double r_eci[3],
    double out_a[3],
    bool use_j2,
    const CentralBodySpecNative& central_body);

// Encke's identity for the two-body differential
//   g_pm(r_chief + rho) - g_pm(r_chief)
//   = -(gm/||r_chief||^3) * (rho - f(sigma) * (r_chief + rho)),
// with sigma = rho . (2 r_chief + rho) / ||r_chief||^2 and
// f(sigma) = 1 - (1+sigma)^{-3/2} evaluated cancellation-free. Avoids the
// catastrophic subtraction of two near-equal large gravity vectors when
// ||rho|| << ||r_chief||.
void encke_point_mass_relative_accel(
    const double rho[3],
    const double r_chief[3],
    double out_a[3],
    double gm = kGmEarth);

// Combined relative gravity g(r_chief + rho) - g(r_chief): Encke for the
// point-mass term, direct subtraction for J2 (which is small enough that
// cancellation is not catastrophic).
void relative_accel(
    const double rho[3],
    const double r_chief[3],
    double out_a[3],
    bool use_j2,
    const CentralBodySpecNative& central_body);

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_GRAVITY_H_
