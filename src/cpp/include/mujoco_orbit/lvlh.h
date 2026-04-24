#ifndef MUJOCO_ORBIT_LVLH_H_
#define MUJOCO_ORBIT_LVLH_H_

#include "mujoco_orbit/orbit_state.h"

namespace mujoco_orbit {

void update_frame_cache(
    const double R_eci[3],
    const double V_eci[3],
    FrameCache* out_cache,
    bool use_j2 = false);

inline void update_frame_cache(const OrbitState& orbit, FrameCache* out_cache, bool use_j2 = false) {
  update_frame_cache(orbit.R_eci, orbit.V_eci, out_cache, use_j2);
}

void eci_to_lvlh_pos(
    const double r_eci[3],
    const double R_ref[3],
    const double C_LI[9],
    double out_r_lvlh[3]);

void lvlh_to_eci_pos(
    const double r_lvlh[3],
    const double R_ref[3],
    const double C_IL[9],
    double out_r_eci[3]);

void eci_to_lvlh_vel(
    const double v_eci[3],
    const double r_lvlh[3],
    const double V_ref[3],
    const double C_LI[9],
    const double omega_lvlh[3],
    double out_v_lvlh[3]);

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_LVLH_H_
