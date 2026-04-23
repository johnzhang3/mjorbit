#ifndef MUJOCO_ORBIT_PLUGIN_ORBIT_INSTANCE_H_
#define MUJOCO_ORBIT_PLUGIN_ORBIT_INSTANCE_H_

// Per-mjData instance state for the mujoco_orbit.orbit plugin.
//
// One OrbitInstance lives behind mjData->plugin_data[instance], allocated in
// init() and freed in destroy(). Each thread that owns an mjData has its own
// OrbitInstance and never shares mutable state with other threads.
//
// This struct is intentionally flat and POD-ish so it is cheap to zero-init
// and trivially copyable for mjData copy semantics (see plugin copy() hook).

namespace mujoco_orbit {

struct SurfaceMetadataNative {
  int body_id;
  double center_of_pressure_body[3];
  double normal_body[3];
  double area;
  double drag_coeff;
  double srp_coeff;
  int use_drag;
  int use_srp;
};

struct MagneticMetadataNative {
  int body_id;
  double dipole_body[3];
};

struct OrbitInstance {
  // Chief orbit state (absolute ECI).
  double R_eci[3];   // km
  double V_eci[3];   // km/s
  double t;          // s since epoch

  // Frame cache (filled by plugin compute()).
  double C_LI[9];           // rotation world-from-LVLH (row-major)
  double C_IL[9];           // rotation LVLH-from-world
  double omega_lvlh[3];     // LVLH angular velocity in world, rad/s
  double omega_dot_lvlh[3]; // LVLH angular acceleration, rad/s^2

  // Environment cache.
  double sun_vector_eci[3];      // unit vector to Sun in ECI
  double mag_field_eci[3];       // magnetic field at chief, T
  double atmosphere_omega_eci[3];  // Earth rotation vector, rad/s
  double atm_density;            // kg/m^3
  double eclipse;                // 0.0 shadow, 1.0 full sun

  // Config (parsed from XML plugin attributes during init()).
  int use_j2;               // include J2 perturbation in chief gravity
  int use_drag;
  int use_srp;
  int use_magnetic;
  int use_gravity_gradient;

  int num_surfaces;
  const SurfaceMetadataNative* surfaces;

  int num_magnetic_bodies;
  const MagneticMetadataNative* magnetic_bodies;
};

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_PLUGIN_ORBIT_INSTANCE_H_
