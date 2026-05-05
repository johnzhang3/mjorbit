// mujoco_orbit.orbit plugin.
//
// The Python shim attaches this plugin instance to a body in every compiled
// model and marshals per-model passive-coupling metadata into the native
// OrbitInstance. Phase 4 uses the PASSIVE callback to apply chief-relative
// gravity, drag/SRP, residual magnetic torque, and gravity-gradient torque.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mujoco.h>

#include "mujoco_orbit/coupling.h"
#include "mujoco_orbit/orbit_cache.h"
#include "mujoco_orbit/orbit_schedule.h"
#include "mujoco_orbit/propagator.h"
#include "mujoco_orbit/sensors_plugin.h"
#include "orbit_instance.h"

namespace {

constexpr char kPluginName[] = "mujoco_orbit.orbit";

constexpr const char* kAttributeNames[] = {"use_j2"};
constexpr int kNumAttributes = sizeof(kAttributeNames) / sizeof(kAttributeNames[0]);

mujoco_orbit::OrbitInstance* GetInstance(mjData* d, int instance) {
  return reinterpret_cast<mujoco_orbit::OrbitInstance*>(d->plugin_data[instance]);
}

int NState(const mjModel* /*m*/, int /*instance*/) {
  // Runtime state is per-mjData plugin_data so threaded rollouts can share one
  // immutable mjModel while keeping independent OrbitInstance objects. If we
  // later need mj_getState/mj_setState to pack orbit state for MuJoCo rollout,
  // this should grow into mjtNum-managed plugin_state instead of adding a
  // separate threading framework.
  return 0;
}

int Init(const mjModel* m, mjData* d, int instance) {
  // value-initialize so std::string members (e.g. central_body.name) are
  // properly constructed; calloc + assignment is UB on a non-trivial type.
  auto* inst = new (std::nothrow) mujoco_orbit::OrbitInstance{};
  if (!inst) return -1;

  // Override the member-initializer defaults that aren't picked up by
  // value-initialization for the int flags.
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

  // Install the global mjcb_sensor hook once, now that the plugin owns at
  // least one instance on this mjData. Doing this here (not at .so load time)
  // avoids Python's MjoModel.from_xml_path stomping our callback via the
  // mjcb_sensor save/restore dance.
  mujoco_orbit::ensure_sensor_callback_installed();
  return 0;
}

void Destroy(mjData* d, int instance) {
  if (auto* inst = GetInstance(d, instance)) {
    delete inst;
    d->plugin_data[instance] = 0;
  }
}

void Copy(mjData* dest, const mjModel* /*m*/, const mjData* src, int instance) {
  auto* src_inst = reinterpret_cast<mujoco_orbit::OrbitInstance*>(src->plugin_data[instance]);
  if (!src_inst) return;
  // Copy-construct so std::string members are properly cloned; raw memcpy of a
  // type containing std::string aliases the SSO buffer and is UB.
  auto* dest_inst = new (std::nothrow) mujoco_orbit::OrbitInstance(*src_inst);
  if (!dest_inst) return;
  dest->plugin_data[instance] = reinterpret_cast<uintptr_t>(dest_inst);
}

void Reset(const mjModel* /*m*/, mjtNum* /*plugin_state*/, void* plugin_data, int /*instance*/) {
  auto* inst = reinterpret_cast<mujoco_orbit::OrbitInstance*>(plugin_data);
  if (!inst) return;
  // Preserve shim-populated config/metadata across mj_resetData. Only the
  // runtime chief orbit + derived caches should be reset to zero. Use move +
  // destructor + placement-new so the std::string in central_body stays valid.
  mujoco_orbit::OrbitInstance preserved = std::move(*inst);
  inst->~OrbitInstance();
  new (inst) mujoco_orbit::OrbitInstance{};
  inst->central_body = std::move(preserved.central_body);
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
  inst->num_orbit_sensors = preserved.num_orbit_sensors;
  inst->orbit_sensors = preserved.orbit_sensors;
}

void Compute(const mjModel* m, mjData* d, int instance, int capability_bit) {
  if (capability_bit != mjPLUGIN_PASSIVE) {
    return;
  }
  auto* inst = GetInstance(d, instance);
  // Caches are kept current by Advance() at the end of the previous step (and
  // by initialize_orbit_schedule on construction / set_orbit / mjo_set_state),
  // so we don't refresh again here on the hot step path.
  mujoco_orbit::apply_passive_wrenches(m, d, inst);
}

void Advance(const mjModel* m, mjData* d, int instance) {
  auto* inst = GetInstance(d, instance);
  if (!inst) return;

  mujoco_orbit::advance_actuators(m, inst);
  mujoco_orbit::advance_orbit_schedule(m, inst);
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
