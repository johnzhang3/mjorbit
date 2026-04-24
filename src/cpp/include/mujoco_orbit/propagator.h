#ifndef MUJOCO_ORBIT_PROPAGATOR_H_
#define MUJOCO_ORBIT_PROPAGATOR_H_

#include "mujoco_orbit/orbit_state.h"

namespace mujoco_orbit {

void propagate_rk4(
    const double R_eci[3],
    const double V_eci[3],
    double t,
    double dt,
    double out_R_eci[3],
    double out_V_eci[3],
    double* out_t,
    bool use_j2 = true,
    const double* a_external = nullptr);

inline void propagate_rk4(
    const OrbitState& state,
    double dt,
    OrbitState* out_state,
    bool use_j2 = true,
    const double* a_external = nullptr) {
  if (!out_state) {
    return;
  }
  propagate_rk4(
      state.R_eci,
      state.V_eci,
      state.t,
      dt,
      out_state->R_eci,
      out_state->V_eci,
      &out_state->t,
      use_j2,
      a_external);
}

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_PROPAGATOR_H_
