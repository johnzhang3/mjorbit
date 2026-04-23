#ifndef MUJOCO_ORBIT_COUPLING_H_
#define MUJOCO_ORBIT_COUPLING_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>

#include "orbit_instance.h"

namespace mujoco_orbit {

void apply_passive_wrenches(const mjModel* m, mjData* d, OrbitInstance* inst);
void advance_actuators(const mjModel* m, OrbitInstance* inst);

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_COUPLING_H_
