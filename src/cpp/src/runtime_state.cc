#include "mujoco_orbit/runtime.h"

#include <cstddef>
#include <stdexcept>

#include "mujoco_orbit/orbit_schedule.h"

extern "C" int mjo_rollout(
    const mjModel* m,
    mjData* d,
    int orbit_plugin_instance,
    int nbatch,
    int nstep,
    unsigned int control_spec,
    int mjo_state_size,
    int mjo_control_size,
    const mjtNum* initial_state,
    const mjtNum* initial_warmstart,
    const mjtNum* control,
    mjtNum* state,
    mjtNum* sensordata);

namespace mujoco_orbit {
namespace {

int mjo_state_tail_size(const MjoModel& model) {
  return 7 + static_cast<int>(model.reaction_wheels().size()) +
         2 * static_cast<int>(model.cmgs().size());
}

int mjo_control_tail_size(const MjoModel& model) {
  return static_cast<int>(
      model.reaction_wheels().size() + model.magnetorquers().size() +
      model.thrusters().size() + model.cmgs().size());
}

void update_reaction_wheel_momentum(MjoData& data) {
  const auto& wheels = data.model().reaction_wheels();
  for (std::size_t i = 0; i < wheels.size(); ++i) {
    data.rw_momentum()[i] = data.rw_speed()[i] * wheels[i].inertia;
  }
}

}  // namespace

void mjo_forward(MjoModel& model, MjoData& data) {
  data.clear_wrench_buffer();
  mj_forward(model.raw(), data.raw());
  mju_copy(data.raw()->xfrc_applied, data.wrench_buffer(), 6 * model.nbody());
}

void mjo_step(MjoModel& model, MjoData& data) {
  data.clear_wrench_buffer();
  mj_step(model.raw(), data.raw());
  mju_copy(data.raw()->xfrc_applied, data.wrench_buffer(), 6 * model.nbody());
}

int mjo_state_size(const MjoModel& model) {
  return mj_stateSize(model.raw(), mjSTATE_FULLPHYSICS) + mjo_state_tail_size(model);
}

int mjo_control_size(const MjoModel& model, unsigned int control_spec) {
  return mj_stateSize(model.raw(), control_spec) + mjo_control_tail_size(model);
}

void mjo_get_state(const MjoModel& model, const MjoData& data, double* out) {
  const int nfull = mj_stateSize(model.raw(), mjSTATE_FULLPHYSICS);
  mj_getState(model.raw(), const_cast<mjData*>(data.raw()), out, mjSTATE_FULLPHYSICS);
  double* tail = out + nfull;
  const OrbitInstance* inst = data.orbit_instance();
  for (int i = 0; i < 3; ++i) *tail++ = inst->R_eci[i];
  for (int i = 0; i < 3; ++i) *tail++ = inst->V_eci[i];
  *tail++ = inst->t;
  for (std::size_t i = 0; i < model.reaction_wheels().size(); ++i) {
    *tail++ = inst->rw_speed ? inst->rw_speed[i] : 0.0;
  }
  for (std::size_t i = 0; i < model.cmgs().size(); ++i) {
    *tail++ = inst->cmg_gimbal_angle ? inst->cmg_gimbal_angle[i] : 0.0;
  }
  for (std::size_t i = 0; i < model.cmgs().size(); ++i) {
    *tail++ = inst->cmg_rotor_momentum ? inst->cmg_rotor_momentum[i] : 0.0;
  }
}

void mjo_set_state(const MjoModel& model, MjoData& data, const double* state, int size) {
  const int expected = mjo_state_size(model);
  if (size != expected) {
    throw std::runtime_error("state has incompatible size");
  }
  const int nfull = mj_stateSize(model.raw(), mjSTATE_FULLPHYSICS);
  mj_setState(model.raw(), data.raw(), state, mjSTATE_FULLPHYSICS);
  const double* tail = state + nfull;
  OrbitInstance* inst = data.orbit_instance();
  for (int i = 0; i < 3; ++i) inst->R_eci[i] = *tail++;
  for (int i = 0; i < 3; ++i) inst->V_eci[i] = *tail++;
  inst->t = *tail++;
  for (std::size_t i = 0; i < model.reaction_wheels().size(); ++i) {
    if (inst->rw_speed) inst->rw_speed[i] = *tail;
    ++tail;
  }
  for (std::size_t i = 0; i < model.cmgs().size(); ++i) {
    if (inst->cmg_gimbal_angle) inst->cmg_gimbal_angle[i] = *tail;
    ++tail;
  }
  for (std::size_t i = 0; i < model.cmgs().size(); ++i) {
    if (inst->cmg_rotor_momentum) inst->cmg_rotor_momentum[i] = *tail;
    ++tail;
  }
  update_reaction_wheel_momentum(data);
  inst->orbit_schedule_initialized = 0;
  initialize_orbit_schedule(model.raw(), inst);
}

int mjo_rollout_native(
    MjoModel& model,
    MjoData& data,
    int nbatch,
    int nstep,
    unsigned int control_spec,
    int state_size,
    int control_size,
    const double* initial_state,
    const double* initial_warmstart,
    const double* control,
    double* state,
    double* sensordata) {
  return mjo_rollout(
      model.raw(),
      data.raw(),
      model.orbit_plugin_instance(),
      nbatch,
      nstep,
      control_spec,
      state_size,
      control_size,
      initial_state,
      initial_warmstart,
      control,
      state,
      sensordata);
}

}  // namespace mujoco_orbit
