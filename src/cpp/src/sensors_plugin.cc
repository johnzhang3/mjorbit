#include "mujoco_orbit/sensors_plugin.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>

#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mujoco.h>

#include "mujoco_orbit/orbit_cache.h"

namespace mujoco_orbit {

namespace {

constexpr char kPluginName[] = "mujoco_orbit.orbit";
constexpr int kPosStage = static_cast<int>(mjSTAGE_POS);

void normalize3(const double in[3], double out[3]) {
  const double n = std::sqrt(in[0] * in[0] + in[1] * in[1] + in[2] * in[2]);
  if (n <= 0.0) {
    out[0] = in[0];
    out[1] = in[1];
    out[2] = in[2];
    return;
  }
  const double inv = 1.0 / n;
  out[0] = in[0] * inv;
  out[1] = in[1] * inv;
  out[2] = in[2] * inv;
}

// site_xmat is row-major world-from-site. The sensor measures the body/site
// frame, so we apply R^T to the world-frame vector.
void site_frame_transform(const double* site_xmat, const double world[3], double out[3]) {
  out[0] = site_xmat[0] * world[0] + site_xmat[3] * world[1] + site_xmat[6] * world[2];
  out[1] = site_xmat[1] * world[0] + site_xmat[4] * world[1] + site_xmat[7] * world[2];
  out[2] = site_xmat[2] * world[0] + site_xmat[5] * world[1] + site_xmat[8] * world[2];
}

int locate_orbit_plugin_instance(const mjModel* m) {
  for (int i = 0; i < m->nplugin; ++i) {
    const mjpPlugin* plugin = mjp_getPluginAtSlot(m->plugin[i]);
    if (plugin && plugin->name && std::strcmp(plugin->name, kPluginName) == 0) {
      return i;
    }
  }
  return -1;
}

std::mutex g_install_mutex;
std::atomic<bool> g_installed{false};
mjfSensor g_previous_callback = nullptr;

void orbit_sensor_callback(const mjModel* m, mjData* d, int stage) {
  if (g_previous_callback) {
    g_previous_callback(m, d, stage);
  }
  const int instance = locate_orbit_plugin_instance(m);
  if (instance < 0) {
    return;
  }
  auto* inst = reinterpret_cast<OrbitInstance*>(d->plugin_data[instance]);
  if (!inst || inst->num_orbit_sensors <= 0) {
    return;
  }
  if (stage == kPosStage) {
    refresh_orbit_caches(inst);
  }
  compute_orbit_sensor_truth(m, d, inst, stage);
}

}  // namespace

void compute_orbit_sensor_truth(
    const mjModel* m,
    mjData* d,
    const OrbitInstance* inst,
    int stage) {
  if (!inst || inst->num_orbit_sensors <= 0 || !inst->orbit_sensors) {
    return;
  }
  if (stage != kPosStage) {
    return;
  }

  for (int i = 0; i < inst->num_orbit_sensors; ++i) {
    const OrbitSensorDescriptorNative& s = inst->orbit_sensors[i];
    if (s.site_id < 0 || s.site_id >= m->nsite) continue;
    if (s.adr < 0 || s.dim <= 0) continue;
    if (s.adr + s.dim > m->nsensordata) continue;

    const double* site_xmat = d->site_xmat + 9 * s.site_id;

    double world_vec[3] = {0.0, 0.0, 0.0};
    bool normalize = true;
    switch (s.kind) {
      case kOrbitSensorSun:
        world_vec[0] = inst->sun_vector_eci[0];
        world_vec[1] = inst->sun_vector_eci[1];
        world_vec[2] = inst->sun_vector_eci[2];
        break;
      case kOrbitSensorHorizon: {
        const double rx = inst->R_eci[0];
        const double ry = inst->R_eci[1];
        const double rz = inst->R_eci[2];
        const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
        if (r <= 0.0) continue;
        const double inv_r = 1.0 / r;
        world_vec[0] = -rx * inv_r;
        world_vec[1] = -ry * inv_r;
        world_vec[2] = -rz * inv_r;
        break;
      }
      case kOrbitSensorStar:
        world_vec[0] = s.reference_eci[0];
        world_vec[1] = s.reference_eci[1];
        world_vec[2] = s.reference_eci[2];
        break;
      case kOrbitSensorMagnetometer:
        world_vec[0] = inst->mag_field_eci[0];
        world_vec[1] = inst->mag_field_eci[1];
        world_vec[2] = inst->mag_field_eci[2];
        // Magnetometer truth is raw field in Tesla, not a unit vector.
        normalize = false;
        break;
      default:
        continue;
    }

    double site_vec[3];
    site_frame_transform(site_xmat, world_vec, site_vec);
    if (normalize) {
      normalize3(site_vec, site_vec);
    }

    d->sensordata[s.adr + 0] = site_vec[0];
    d->sensordata[s.adr + 1] = site_vec[1];
    d->sensordata[s.adr + 2] = site_vec[2];
  }
}

void ensure_sensor_callback_installed() {
  if (g_installed.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(g_install_mutex);
  if (g_installed.load(std::memory_order_relaxed)) return;

  if (mjcb_sensor != orbit_sensor_callback) {
    g_previous_callback = mjcb_sensor;
    mjcb_sensor = orbit_sensor_callback;
  }
  g_installed.store(true, std::memory_order_release);
}

}  // namespace mujoco_orbit
