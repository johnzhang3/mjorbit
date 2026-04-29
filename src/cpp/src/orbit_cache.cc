#include "mujoco_orbit/orbit_cache.h"

#include <cstring>

#include "mujoco_orbit/environment.h"
#include "mujoco_orbit/lvlh.h"

namespace mujoco_orbit {

void refresh_orbit_caches(OrbitInstance* inst) {
  if (!inst) {
    return;
  }

  OrbitState orbit{};
  std::memcpy(orbit.R_eci, inst->R_eci, sizeof(orbit.R_eci));
  std::memcpy(orbit.V_eci, inst->V_eci, sizeof(orbit.V_eci));
  orbit.t = inst->t;

  FrameCache frame{};
  update_frame_cache(orbit, &frame, inst->use_j2 != 0, inst->central_body);
  std::memcpy(inst->C_LI, frame.C_LI, sizeof(inst->C_LI));
  std::memcpy(inst->C_IL, frame.C_IL, sizeof(inst->C_IL));
  std::memcpy(inst->omega_lvlh, frame.omega_lvlh, sizeof(inst->omega_lvlh));
  std::memcpy(inst->omega_dot_lvlh, frame.omega_dot_lvlh, sizeof(inst->omega_dot_lvlh));

  EnvironmentCache env{};
  update_environment_cache(orbit, frame, &env, inst->central_body);
  std::memcpy(inst->sun_vector_eci, env.sun_vector_eci, sizeof(inst->sun_vector_eci));
  std::memcpy(inst->mag_field_eci, env.mag_field_eci, sizeof(inst->mag_field_eci));
  std::memcpy(
      inst->atmosphere_omega_eci,
      env.atmosphere_omega_eci,
      sizeof(inst->atmosphere_omega_eci));
  inst->atm_density = env.atm_density;
  inst->eclipse = env.eclipse;
}

}  // namespace mujoco_orbit
