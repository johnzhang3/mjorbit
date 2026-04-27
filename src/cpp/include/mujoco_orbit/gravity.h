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

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_GRAVITY_H_
