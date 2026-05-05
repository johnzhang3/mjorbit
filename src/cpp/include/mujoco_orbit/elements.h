#ifndef MUJOCO_ORBIT_ELEMENTS_H_
#define MUJOCO_ORBIT_ELEMENTS_H_

#include "mujoco_orbit/constants.h"

namespace mujoco_orbit {

void keplerian_to_cartesian(
    double a,
    double e,
    double inc,
    double raan,
    double argp,
    double nu,
    double out_R_eci[3],
    double out_V_eci[3],
    double gm = kGmEarth);

void cartesian_to_keplerian(
    const double R_eci[3],
    const double V_eci[3],
    double* out_a,
    double* out_e,
    double* out_inc,
    double* out_raan,
    double* out_argp,
    double* out_nu,
    double gm = kGmEarth);

void rot_pf_to_eci(double raan, double inc, double argp, double out_R[9]);

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_ELEMENTS_H_
