#include "mujoco_orbit/orbit_schedule.h"

#include <algorithm>

#include "mujoco_orbit/math_utils.h"
#include "mujoco_orbit/orbit_cache.h"
#include "mujoco_orbit/propagator.h"

namespace mujoco_orbit {
namespace {

constexpr double kEps = 1.0e-12;

void copy_state_to_segment_start(OrbitInstance* inst) {
  detail::copy3(inst->R_eci, inst->orbit_segment_start_R_eci);
  detail::copy3(inst->V_eci, inst->orbit_segment_start_V_eci);
  inst->orbit_segment_start_t = inst->t;
}

void clear_feedback_integral(OrbitInstance* inst) {
  detail::zero3(inst->feedback_accel_integral_eci);
  inst->feedback_accel_integral_dt = 0.0;
}

double configured_orbit_dt(const mjModel* m, const OrbitInstance* inst) {
  const double dt = inst->orbit_dt > 0.0 ? inst->orbit_dt : m->opt.timestep;
  return std::max(dt, kEps);
}

void average_feedback_accel(const OrbitInstance* inst, double out[3]) {
  if (inst->feedback_accel_integral_dt <= kEps) {
    detail::copy3(inst->feedback_accel_eci, out);
    return;
  }
  detail::scale3(
      inst->feedback_accel_integral_eci,
      1.0 / inst->feedback_accel_integral_dt,
      out);
}

void predict_segment_end(const mjModel* m, OrbitInstance* inst, const double accel[3]) {
  propagate_rk4(
      inst->orbit_segment_start_R_eci,
      inst->orbit_segment_start_V_eci,
      inst->orbit_segment_start_t,
      inst->orbit_segment_duration,
      inst->orbit_segment_end_R_eci,
      inst->orbit_segment_end_V_eci,
      &inst->orbit_segment_end_t,
      inst->use_j2 != 0,
      accel);
  (void) m;
  ++inst->orbit_rk4_count;
}

void begin_segment(const mjModel* m, OrbitInstance* inst, double duration) {
  copy_state_to_segment_start(inst);
  inst->orbit_segment_duration = std::max(duration, kEps);
  inst->orbit_segment_elapsed = 0.0;
  clear_feedback_integral(inst);
  predict_segment_end(m, inst, inst->feedback_accel_eci);
}

void interpolate_segment(OrbitInstance* inst) {
  const double alpha = detail::clamp(
      inst->orbit_segment_elapsed / std::max(inst->orbit_segment_duration, kEps),
      0.0,
      1.0);
  for (int i = 0; i < 3; ++i) {
    inst->R_eci[i] = (1.0 - alpha) * inst->orbit_segment_start_R_eci[i] +
                     alpha * inst->orbit_segment_end_R_eci[i];
    inst->V_eci[i] = (1.0 - alpha) * inst->orbit_segment_start_V_eci[i] +
                     alpha * inst->orbit_segment_end_V_eci[i];
  }
  inst->t = inst->orbit_segment_start_t + inst->orbit_segment_elapsed;
}

void commit_segment_with_average_feedback(const mjModel* m, OrbitInstance* inst) {
  double avg_accel[3];
  average_feedback_accel(inst, avg_accel);
  predict_segment_end(m, inst, avg_accel);
  detail::copy3(inst->orbit_segment_end_R_eci, inst->R_eci);
  detail::copy3(inst->orbit_segment_end_V_eci, inst->V_eci);
  inst->t = inst->orbit_segment_end_t;
}

void accumulate_feedback(OrbitInstance* inst, double dt) {
  for (int i = 0; i < 3; ++i) {
    inst->feedback_accel_integral_eci[i] += inst->feedback_accel_eci[i] * dt;
  }
  inst->feedback_accel_integral_dt += dt;
}

void advance_with_substeps(const mjModel* m, OrbitInstance* inst, double dt, double max_substep) {
  double remaining = dt;
  while (remaining > kEps) {
    const double sub_dt = std::min(remaining, max_substep);
    propagate_rk4(
        inst->R_eci,
        inst->V_eci,
        inst->t,
        sub_dt,
        inst->R_eci,
        inst->V_eci,
        &inst->t,
        inst->use_j2 != 0,
        inst->feedback_accel_eci);
    ++inst->orbit_rk4_count;
    remaining -= sub_dt;
  }
}

}  // namespace

void initialize_orbit_schedule(const mjModel* m, OrbitInstance* inst) {
  if (!m || !inst) {
    return;
  }
  inst->orbit_schedule_initialized = 1;
  inst->orbit_segment_duration = configured_orbit_dt(m, inst);
  inst->orbit_segment_elapsed = 0.0;
  clear_feedback_integral(inst);
  begin_segment(m, inst, inst->orbit_segment_duration);
  detail::copy3(inst->orbit_segment_start_R_eci, inst->R_eci);
  detail::copy3(inst->orbit_segment_start_V_eci, inst->V_eci);
  inst->t = inst->orbit_segment_start_t;
  refresh_orbit_caches(inst);
}

void advance_orbit_schedule(const mjModel* m, OrbitInstance* inst) {
  if (!m || !inst) {
    return;
  }

  const double mj_dt = m->opt.timestep;
  const double orbit_dt = configured_orbit_dt(m, inst);
  if (!inst->orbit_schedule_initialized) {
    initialize_orbit_schedule(m, inst);
  }

  if (orbit_dt <= mj_dt + kEps) {
    advance_with_substeps(m, inst, mj_dt, orbit_dt);
    copy_state_to_segment_start(inst);
    inst->orbit_segment_duration = orbit_dt;
    inst->orbit_segment_elapsed = 0.0;
    clear_feedback_integral(inst);
    refresh_orbit_caches(inst);
    return;
  }

  double remaining = mj_dt;
  while (remaining > kEps) {
    const double segment_remaining = inst->orbit_segment_duration - inst->orbit_segment_elapsed;
    const double dt = std::min(remaining, segment_remaining);
    accumulate_feedback(inst, dt);
    inst->orbit_segment_elapsed += dt;
    remaining -= dt;

    if (inst->orbit_segment_elapsed + kEps >= inst->orbit_segment_duration) {
      commit_segment_with_average_feedback(m, inst);
      begin_segment(m, inst, orbit_dt);
    } else {
      interpolate_segment(inst);
    }
  }

  refresh_orbit_caches(inst);
}

}  // namespace mujoco_orbit
