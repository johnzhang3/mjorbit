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

void Reset(const mjModel* /*m*/, mjtNum* /*plugin_state*/, void* plugin_data, int /*instance*/) {
  auto* inst = reinterpret_cast<mujoco_orbit::OrbitInstance*>(plugin_data);
  if (!inst) return;
  // Preserve shim-populated config/metadata across mj_resetData. Only the
  // runtime chief orbit + derived caches should be reset to zero.
  const int use_j2 = inst->use_j2;
  const int use_drag = inst->use_drag;
  const int use_srp = inst->use_srp;
  const int use_magnetic = inst->use_magnetic;
  const int use_gravity_gradient = inst->use_gravity_gradient;
  const int num_surfaces = inst->num_surfaces;
  const auto* surfaces = inst->surfaces;
  const int num_magnetic_bodies = inst->num_magnetic_bodies;
  const auto* magnetic_bodies = inst->magnetic_bodies;
  std::memset(inst, 0, sizeof(*inst));
  inst->use_j2 = use_j2;
  inst->use_drag = use_drag;
  inst->use_srp = use_srp;
  inst->use_magnetic = use_magnetic;
  inst->use_gravity_gradient = use_gravity_gradient;
  inst->num_surfaces = num_surfaces;
  inst->surfaces = surfaces;
  inst->num_magnetic_bodies = num_magnetic_bodies;
  inst->magnetic_bodies = magnetic_bodies;
}

void Compute(const mjModel* m, mjData* d, int instance, int capability_bit) {
  if (capability_bit != mjPLUGIN_PASSIVE) {
    return;
  }
  mujoco_orbit::apply_passive_wrenches(m, d, GetInstance(d, instance));
}

void Advance(const mjModel* /*m*/, mjData* /*d*/, int /*instance*/) {
  // Phase 1 stub: no orbit propagation. Phase 5 fills this in.
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
