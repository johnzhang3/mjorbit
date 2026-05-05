#ifndef MUJOCO_ORBIT_ORBIT_STATE_H_
#define MUJOCO_ORBIT_ORBIT_STATE_H_

namespace mujoco_orbit {

struct OrbitState {
  double R_eci[3];
  double V_eci[3];
  double t;
};

struct FrameCache {
  double C_LI[9];
  double C_IL[9];
  double omega_lvlh[3];
  double omega_dot_lvlh[3];
};

struct EnvironmentCache {
  double sun_vector_eci[3];
  double eclipse;
  double mag_field_eci[3];
  double atmosphere_omega_eci[3];
  double atm_density;
};

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_ORBIT_STATE_H_
