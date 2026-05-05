#ifndef MUJOCO_ORBIT_ORBIT_SCHEDULE_H_
#define MUJOCO_ORBIT_ORBIT_SCHEDULE_H_

#include <mujoco/mjmodel.h>

#include "orbit_instance.h"

namespace mujoco_orbit {

void initialize_orbit_schedule(const mjModel* m, OrbitInstance* inst);
void advance_orbit_schedule(const mjModel* m, OrbitInstance* inst);

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_ORBIT_SCHEDULE_H_
