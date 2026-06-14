#ifndef MJORBIT_ORBIT_SCHEDULE_H_
#define MJORBIT_ORBIT_SCHEDULE_H_

#include <mujoco/mjmodel.h>

#include "orbit_instance.h"

namespace mjorbit {

void initialize_orbit_schedule(const mjModel* m, OrbitInstance* inst);
void advance_orbit_schedule(const mjModel* m, OrbitInstance* inst);

}  // namespace mjorbit

#endif  // MJORBIT_ORBIT_SCHEDULE_H_
