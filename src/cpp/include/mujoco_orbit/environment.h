#ifndef MUJOCO_ORBIT_ENVIRONMENT_H_
#define MUJOCO_ORBIT_ENVIRONMENT_H_

#include "mujoco_orbit/orbit_state.h"

namespace mujoco_orbit {

void sun_vector_eci(double t, double out_sun_hat[3]);

double eclipse_factor(const double R_eci[3], const double sun_hat[3]);

void dipole_field_eci(const double R_eci[3], double t, double out_B_eci[3]);

double atm_density(const double R_eci[3]);

void atmosphere_relative_velocity_eci(
    const double V_sc_eci[3],
    const double R_eci[3],
    double out_v_rel_eci[3]);

void update_environment_cache(
    const OrbitState& orbit,
    const FrameCache& frame_cache,
    EnvironmentCache* out_cache);

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_ENVIRONMENT_H_
