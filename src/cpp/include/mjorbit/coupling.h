#ifndef MJORBIT_COUPLING_H_
#define MJORBIT_COUPLING_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>

#include "orbit_instance.h"

namespace mjorbit {

void apply_passive_wrenches(const mjModel* m, mjData* d, OrbitInstance* inst);
void advance_actuators(const mjModel* m, OrbitInstance* inst);

}  // namespace mjorbit

#endif  // MJORBIT_COUPLING_H_
