#ifndef MUJOCO_ORBIT_SENSORS_PLUGIN_H_
#define MUJOCO_ORBIT_SENSORS_PLUGIN_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>

#include "orbit_instance.h"

namespace mujoco_orbit {

constexpr int kOrbitSensorSun = 1;
constexpr int kOrbitSensorHorizon = 2;
constexpr int kOrbitSensorStar = 3;
constexpr int kOrbitSensorMagnetometer = 4;

// Overwrite sensordata for orbit-owned sensors at the pos stage.
// Safe to call with inst == nullptr or an empty sensor catalog.
void compute_orbit_sensor_truth(
    const mjModel* m,
    mjData* d,
    const OrbitInstance* inst,
    int stage);

// Install a global mjcb_sensor hook that locates the mujoco_orbit.orbit plugin
// instance in `m` and dispatches to compute_orbit_sensor_truth. Idempotent —
// the first successful install chains to any pre-existing callback.
void ensure_sensor_callback_installed();

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_SENSORS_PLUGIN_H_
