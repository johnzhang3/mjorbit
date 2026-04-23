// mujoco_orbit.orbit plugin.
//
// The Python shim attaches this plugin instance to a body in every compiled
// model and marshals per-model passive-coupling metadata into the native
// OrbitInstance. Phase 4 uses the PASSIVE callback to apply chief-relative
// gravity, drag/SRP, residual magnetic torque, and gravity-gradient torque.

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mujoco.h>

#include "mujoco_orbit/coupling.h"
#include "mujoco_orbit/environment.h"
#include "mujoco_orbit/lvlh.h"
#include "mujoco_orbit/propagator.h"
#include "orbit_instance.h"

namespace {

constexpr char kPluginName[] = "mujoco_orbit.orbit";

constexpr const char* kAttributeNames[] = {"use_j2"};
constexpr int kNumAttributes = sizeof(kAttributeNames) / sizeof(kAttributeNames[0]);

mujoco_orbit::OrbitInstance* GetInstance(mjData* d, int instance) {
  return reinterpret_cast<mujoco_orbit::OrbitInstance*>(d->plugin_data[instance]);
}

int NState(const mjModel* /*m*/, int /*instance*/) {
  // No mjtNum-managed state; we hold everything behind plugin_data.
  return 0;
}

int Init(const mjModel* m, mjData* d, int instance) {
  auto* inst = static_cast<mujoco_orbit::OrbitInstance*>(
      std::calloc(1, sizeof(mujoco_orbit::OrbitInstance)));
  if (!inst) return -1;

  // Defaults — Python shim / XML attributes override these later.
  inst->use_j2 = 1;
  inst->use_drag = 1;
  inst->use_srp = 1;
  inst->use_magnetic = 1;
  inst->use_gravity_gradient = 1;

  const char* use_j2_attr = mj_getPluginConfig(m, instance, "use_j2");
  if (use_j2_attr && use_j2_attr[0] != '\0') {
    inst->use_j2 = (std::strcmp(use_j2_attr, "false") == 0 ||
                    std::strcmp(use_j2_attr, "0") == 0) ? 0 : 1;
  }

  d->plugin_data[instance] = reinterpret_cast<uintptr_t>(inst);
  return 0;
}

void Destroy(mjData* d, int instance) {
  if (auto* inst = GetInstance(d, instance)) {
    std::free(inst);
    d->plugin_data[instance] = 0;
  }
}

void Copy(mjData* dest, const mjModel* /*m*/, const mjData* src, int instance) {
  auto* src_inst = reinterpret_cast<mujoco_orbit::OrbitInstance*>(src->plugin_data[instance]);
  if (!src_inst) return;
  auto* dest_inst = static_cast<mujoco_orbit::OrbitInstance*>(
      std::calloc(1, sizeof(mujoco_orbit::OrbitInstance)));
  if (!dest_inst) return;
  std::memcpy(dest_inst, src_inst, sizeof(mujoco_orbit::OrbitInstance));
  dest->plugin_data[instance] = reinterpret_cast<uintptr_t>(dest_inst);
}

void RefreshCaches(mujoco_orbit::OrbitInstance* inst) {
  mujoco_orbit::OrbitState orbit{};
  std::memcpy(orbit.R_eci, inst->R_eci, sizeof(orbit.R_eci));
  std::memcpy(orbit.V_eci, inst->V_eci, sizeof(orbit.V_eci));
  orbit.t = inst->t;

  mujoco_orbit::FrameCache frame{};
  mujoco_orbit::update_frame_cache(orbit, &frame, inst->use_j2 != 0);
  std::memcpy(inst->C_LI, frame.C_LI, sizeof(inst->C_LI));
  std::memcpy(inst->C_IL, frame.C_IL, sizeof(inst->C_IL));
  std::memcpy(inst->omega_lvlh, frame.omega_lvlh, sizeof(inst->omega_lvlh));
  std::memcpy(inst->omega_dot_lvlh, frame.omega_dot_lvlh, sizeof(inst->omega_dot_lvlh));

  mujoco_orbit::EnvironmentCache env{};
  mujoco_orbit::update_environment_cache(orbit, frame, &env);
  std::memcpy(inst->sun_vector_eci, env.sun_vector_eci, sizeof(inst->sun_vector_eci));
  std::memcpy(inst->mag_field_eci, env.mag_field_eci, sizeof(inst->mag_field_eci));
  std::memcpy(
      inst->atmosphere_omega_eci,
      env.atmosphere_omega_eci,
      sizeof(inst->atmosphere_omega_eci));
  inst->atm_density = env.atm_density;
  inst->eclipse = env.eclipse;
}

void Reset(const mjModel* /*m*/, mjtNum* /*plugin_state*/, void* plugin_data, int /*instance*/) {
  auto* inst = reinterpret_cast<mujoco_orbit::OrbitInstance*>(plugin_data);
  if (!inst) return;
  // Preserve shim-populated config/metadata across mj_resetData. Only the
  // runtime chief orbit + derived caches should be reset to zero.
  const auto preserved = *inst;
  std::memset(inst, 0, sizeof(*inst));
  inst->use_j2 = preserved.use_j2;
  inst->use_drag = preserved.use_drag;
  inst->use_srp = preserved.use_srp;
  inst->use_magnetic = preserved.use_magnetic;
  inst->use_gravity_gradient = preserved.use_gravity_gradient;
  inst->orbit_dt = preserved.orbit_dt;
  inst->num_surfaces = preserved.num_surfaces;
  inst->surfaces = preserved.surfaces;
  inst->num_magnetic_bodies = preserved.num_magnetic_bodies;
  inst->magnetic_bodies = preserved.magnetic_bodies;
  inst->num_reaction_wheels = preserved.num_reaction_wheels;
  inst->reaction_wheels = preserved.reaction_wheels;
  inst->rw_speed = preserved.rw_speed;
  inst->rw_momentum = preserved.rw_momentum;
  inst->rw_torque_cmd = preserved.rw_torque_cmd;
  inst->num_magnetorquers = preserved.num_magnetorquers;
  inst->magnetorquers = preserved.magnetorquers;
  inst->mtq_dipole_cmd = preserved.mtq_dipole_cmd;
  inst->num_thrusters = preserved.num_thrusters;
  inst->thrusters = preserved.thrusters;
  inst->thr_force_cmd = preserved.thr_force_cmd;
  inst->num_cmgs = preserved.num_cmgs;
  inst->cmgs = preserved.cmgs;
  inst->cmg_gimbal_angle = preserved.cmg_gimbal_angle;
  inst->cmg_gimbal_rate_cmd = preserved.cmg_gimbal_rate_cmd;
  inst->cmg_rotor_momentum = preserved.cmg_rotor_momentum;
  inst->wrench_buffer = preserved.wrench_buffer;
  inst->wrench_body_count = preserved.wrench_body_count;
}

void Compute(const mjModel* m, mjData* d, int instance, int capability_bit) {
  if (capability_bit != mjPLUGIN_PASSIVE) {
    return;
  }
  mujoco_orbit::apply_passive_wrenches(m, d, GetInstance(d, instance));
}

void Advance(const mjModel* m, mjData* d, int instance) {
  auto* inst = GetInstance(d, instance);
  if (!inst) return;

  mujoco_orbit::advance_actuators(m, inst);

  const double dt = inst->orbit_dt > 0.0 ? inst->orbit_dt : m->opt.timestep;
  mujoco_orbit::propagate_rk4(
      inst->R_eci,
      inst->V_eci,
      inst->t,
      dt,
      inst->R_eci,
      inst->V_eci,
      &inst->t,
      inst->use_j2 != 0,
      inst->feedback_accel_eci);
  RefreshCaches(inst);
}

void RegisterPlugin() {
  mjpPlugin plugin;
  std::memset(&plugin, 0, sizeof(plugin));
  plugin.name = kPluginName;
  plugin.nattribute = kNumAttributes;
  plugin.attributes = kAttributeNames;
  plugin.capabilityflags = mjPLUGIN_PASSIVE;
  plugin.needstage = mjSTAGE_ACC;
  plugin.nstate = NState;
  plugin.init = Init;
  plugin.destroy = Destroy;
  plugin.copy = Copy;
  plugin.reset = Reset;
  plugin.compute = Compute;
  plugin.advance = Advance;
  mjp_registerPlugin(&plugin);
}

}  // namespace

// Register at .so load time.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
static void _mujoco_orbit_plugin_init() { RegisterPlugin(); }
#elif defined(_MSC_VER)
extern "C" int __stdcall DllMain(void*, unsigned long reason, void*) {
  if (reason == 1) RegisterPlugin();
  return 1;
}
#else
#error "Unsupported compiler: plugin library init hook unavailable"
#endif
