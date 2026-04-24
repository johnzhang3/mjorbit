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

struct ReactionWheelMetadataNative {
  int body_id;
  double axis_body[3];
  double inertia;
  double speed_limit;
  double torque_limit;
  int has_speed_limit;
  int has_torque_limit;
};

struct MagnetorquerMetadataNative {
  int body_id;
  double axis_body[3];
  double dipole_limit;
};

struct ControlMomentGyroMetadataNative {
  int body_id;
  double gimbal_axis_body[3];
  double spin_axis_body_0[3];
  double torque_axis_body_0[3];
  double rotor_momentum;
  double gimbal_rate_limit;
  double gimbal_angle_limit;
  int has_gimbal_rate_limit;
  int has_gimbal_angle_limit;
};

struct ThrusterMetadataNative {
  int body_id;
  double position_body[3];
  double direction_body[3];
  double force_limit;
};

// Per-sensor descriptor for plugin-owned truth generation.
//
// kind:
//   1 = orbit sun vector  (site-frame unit vector to the Sun)
//   2 = orbit horizon vec (site-frame unit vector to nadir)
//   3 = orbit star tracker (site-frame unit vector to a fixed ECI reference)
//   4 = magnetometer      (site-frame chief magnetic field, T)
// site_id indexes mjModel.site_xmat for the frame transform.
// reference_eci is only used for star-tracker sensors (pre-normalised).
struct OrbitSensorDescriptorNative {
  int sensor_id;
  int kind;
  int site_id;
  int adr;
  int dim;
  double reference_eci[3];
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

  // Non-gravitational translational loads accumulated by passive(), N.
  double feedback_force_world[3];
  // Last chief-orbit feedback acceleration computed by passive(), km/s^2.
  double feedback_accel_eci[3];

  // Config (parsed from XML plugin attributes during init()).
  int use_j2;               // include J2 perturbation in chief gravity
  int use_drag;
  int use_srp;
  int use_magnetic;
  int use_gravity_gradient;
  double orbit_dt;           // s; <= 0 uses mjModel.opt.timestep

  int num_surfaces;
  const SurfaceMetadataNative* surfaces;

  int num_magnetic_bodies;
  const MagneticMetadataNative* magnetic_bodies;

  int num_reaction_wheels;
  const ReactionWheelMetadataNative* reaction_wheels;
  double* rw_speed;
  double* rw_momentum;
  double* rw_torque_cmd;

  int num_magnetorquers;
  const MagnetorquerMetadataNative* magnetorquers;
  double* mtq_dipole_cmd;

  int num_thrusters;
  const ThrusterMetadataNative* thrusters;
  double* thr_force_cmd;

  int num_cmgs;
  const ControlMomentGyroMetadataNative* cmgs;
  double* cmg_gimbal_angle;
  double* cmg_gimbal_rate_cmd;
  double* cmg_rotor_momentum;

  // Optional Python-facing wrench snapshot, shape (wrench_body_count, 6).
  double* wrench_buffer;
  int wrench_body_count;

  int num_orbit_sensors;
  const OrbitSensorDescriptorNative* orbit_sensors;
};

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_PLUGIN_ORBIT_INSTANCE_H_
