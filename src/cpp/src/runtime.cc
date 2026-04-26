#include "mujoco_orbit/runtime.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "mujoco_orbit/math_utils.h"
#include "mujoco_orbit/orbit_schedule.h"

namespace mujoco_orbit {

MjoData::MjoData(
    MjoModel& model,
    const std::array<double, 3>& R_eci,
    const std::array<double, 3>& V_eci,
    double t,
    std::optional<std::uint64_t> rng_seed)
    : model_(&model) {
  data_ = mj_makeData(model_->raw());
  if (!data_) {
    throw std::runtime_error("mj_makeData failed");
  }
  orbit_instance_ = reinterpret_cast<OrbitInstance*>(
      data_->plugin_data[model_->orbit_plugin_instance()]);
  if (!orbit_instance_) {
    throw std::runtime_error("mujoco_orbit plugin instance is not initialized");
  }

  rw_speed_.assign(model_->reaction_wheels().size(), 0.0);
  rw_momentum_.assign(model_->reaction_wheels().size(), 0.0);
  rw_torque_cmd_.assign(model_->reaction_wheels().size(), 0.0);
  mtq_dipole_cmd_.assign(model_->magnetorquers().size(), 0.0);
  thr_force_cmd_.assign(model_->thrusters().size(), 0.0);
  cmg_gimbal_angle_.assign(model_->cmgs().size(), 0.0);
  cmg_gimbal_rate_cmd_.assign(model_->cmgs().size(), 0.0);
  cmg_rotor_momentum_.reserve(model_->cmgs().size());
  for (const ControlMomentGyroMetadataNative& cmg : model_->cmgs()) {
    cmg_rotor_momentum_.push_back(cmg.rotor_momentum);
  }
  wrench_buffer_.assign(static_cast<std::size_t>(model_->nbody()) * 6, 0.0);
  bind_native_metadata();
  set_orbit(R_eci, V_eci, t);
  initialize_sensor_biases(rng_seed);
  mjo_forward(*model_, *this);
}

MjoData::~MjoData() {
  if (data_) {
    mj_deleteData(data_);
    data_ = nullptr;
  }
}

void MjoData::bind_native_metadata() {
  OrbitInstance* inst = orbit_instance_;
  inst->use_j2 = model_->use_j2() ? 1 : 0;
  inst->use_drag = model_->use_drag() ? 1 : 0;
  inst->use_srp = model_->use_srp() ? 1 : 0;
  inst->use_magnetic = model_->use_magnetic() ? 1 : 0;
  inst->use_gravity_gradient = model_->use_gravity_gradient() ? 1 : 0;
  inst->orbit_dt = model_->orbit_dt();

  inst->num_surfaces = static_cast<int>(model_->surfaces().size());
  inst->surfaces = model_->surfaces().empty() ? nullptr : model_->surfaces().data();
  inst->num_magnetic_bodies = static_cast<int>(model_->magnetic_bodies().size());
  inst->magnetic_bodies =
      model_->magnetic_bodies().empty() ? nullptr : model_->magnetic_bodies().data();
  inst->num_reaction_wheels = static_cast<int>(model_->reaction_wheels().size());
  inst->reaction_wheels =
      model_->reaction_wheels().empty() ? nullptr : model_->reaction_wheels().data();
  inst->rw_speed = rw_speed_.empty() ? nullptr : rw_speed_.data();
  inst->rw_momentum = rw_momentum_.empty() ? nullptr : rw_momentum_.data();
  inst->rw_torque_cmd = rw_torque_cmd_.empty() ? nullptr : rw_torque_cmd_.data();
  inst->num_magnetorquers = static_cast<int>(model_->magnetorquers().size());
  inst->magnetorquers =
      model_->magnetorquers().empty() ? nullptr : model_->magnetorquers().data();
  inst->mtq_dipole_cmd = mtq_dipole_cmd_.empty() ? nullptr : mtq_dipole_cmd_.data();
  inst->num_thrusters = static_cast<int>(model_->thrusters().size());
  inst->thrusters = model_->thrusters().empty() ? nullptr : model_->thrusters().data();
  inst->thr_force_cmd = thr_force_cmd_.empty() ? nullptr : thr_force_cmd_.data();
  inst->num_cmgs = static_cast<int>(model_->cmgs().size());
  inst->cmgs = model_->cmgs().empty() ? nullptr : model_->cmgs().data();
  inst->cmg_gimbal_angle = cmg_gimbal_angle_.empty() ? nullptr : cmg_gimbal_angle_.data();
  inst->cmg_gimbal_rate_cmd =
      cmg_gimbal_rate_cmd_.empty() ? nullptr : cmg_gimbal_rate_cmd_.data();
  inst->cmg_rotor_momentum =
      cmg_rotor_momentum_.empty() ? nullptr : cmg_rotor_momentum_.data();
  inst->wrench_buffer = wrench_buffer_.empty() ? nullptr : wrench_buffer_.data();
  inst->wrench_body_count = model_->nbody();
  inst->num_orbit_sensors = static_cast<int>(model_->orbit_sensors().size());
  inst->orbit_sensors =
      model_->orbit_sensors().empty() ? nullptr : model_->orbit_sensors().data();
}

void MjoData::set_orbit(
    const std::array<double, 3>& R_eci,
    const std::array<double, 3>& V_eci,
    double t) {
  for (int i = 0; i < 3; ++i) {
    orbit_instance_->R_eci[i] = R_eci[static_cast<std::size_t>(i)];
    orbit_instance_->V_eci[i] = V_eci[static_cast<std::size_t>(i)];
  }
  orbit_instance_->t = t;
  orbit_instance_->orbit_schedule_initialized = 0;
  initialize_orbit_schedule(model_->raw(), orbit_instance_);
}

void MjoData::reset(
    const std::optional<std::array<double, 3>>& R_eci,
    const std::optional<std::array<double, 3>>& V_eci,
    std::optional<double> t) {
  std::array<double, 3> R = {orbit_instance_->R_eci[0], orbit_instance_->R_eci[1], orbit_instance_->R_eci[2]};
  std::array<double, 3> V = {orbit_instance_->V_eci[0], orbit_instance_->V_eci[1], orbit_instance_->V_eci[2]};
  const double time = t.value_or(orbit_instance_->t);
  if (R_eci.has_value()) R = *R_eci;
  if (V_eci.has_value()) V = *V_eci;
  mj_resetData(model_->raw(), data_);
  orbit_instance_ = reinterpret_cast<OrbitInstance*>(
      data_->plugin_data[model_->orbit_plugin_instance()]);
  bind_native_metadata();
  std::fill(rw_speed_.begin(), rw_speed_.end(), 0.0);
  std::fill(rw_momentum_.begin(), rw_momentum_.end(), 0.0);
  std::fill(rw_torque_cmd_.begin(), rw_torque_cmd_.end(), 0.0);
  std::fill(mtq_dipole_cmd_.begin(), mtq_dipole_cmd_.end(), 0.0);
  std::fill(thr_force_cmd_.begin(), thr_force_cmd_.end(), 0.0);
  std::fill(cmg_gimbal_angle_.begin(), cmg_gimbal_angle_.end(), 0.0);
  std::fill(cmg_gimbal_rate_cmd_.begin(), cmg_gimbal_rate_cmd_.end(), 0.0);
  for (std::size_t i = 0; i < model_->cmgs().size(); ++i) {
    cmg_rotor_momentum_[i] = model_->cmgs()[i].rotor_momentum;
  }
  set_orbit(R, V, time);
  mjo_forward(*model_, *this);
}

void MjoData::clear_wrench_buffer() {
  std::fill(wrench_buffer_.begin(), wrench_buffer_.end(), 0.0);
  if (data_) {
    mju_zero(data_->xfrc_applied, 6 * model_->nbody());
  }
}

std::array<double, 3> MjoData::world_position_from_lvlh(
    const std::array<double, 3>& position_lvlh_m) const {
  const double position_lvlh_km[3] = {
      position_lvlh_m[0] * 1.0e-3,
      position_lvlh_m[1] * 1.0e-3,
      position_lvlh_m[2] * 1.0e-3};
  double world_km[3];
  detail::mat3_mul_vec(orbit_instance_->C_IL, position_lvlh_km, world_km);
  return {1000.0 * world_km[0], 1000.0 * world_km[1], 1000.0 * world_km[2]};
}

std::array<double, 3> MjoData::world_velocity_from_lvlh(
    const std::array<double, 3>& position_lvlh_m,
    const std::array<double, 3>& velocity_lvlh_m_s) const {
  const double position_lvlh_km[3] = {
      position_lvlh_m[0] * 1.0e-3,
      position_lvlh_m[1] * 1.0e-3,
      position_lvlh_m[2] * 1.0e-3};
  const double velocity_lvlh_km_s[3] = {
      velocity_lvlh_m_s[0] * 1.0e-3,
      velocity_lvlh_m_s[1] * 1.0e-3,
      velocity_lvlh_m_s[2] * 1.0e-3};
  double omega_cross_r[3];
  detail::cross3(orbit_instance_->omega_lvlh, position_lvlh_km, omega_cross_r);
  double lvlh_total[3];
  detail::add3(velocity_lvlh_km_s, omega_cross_r, lvlh_total);
  double world_km_s[3];
  detail::mat3_mul_vec(orbit_instance_->C_IL, lvlh_total, world_km_s);
  return {1000.0 * world_km_s[0], 1000.0 * world_km_s[1], 1000.0 * world_km_s[2]};
}

std::array<double, 3> MjoData::lvlh_position_from_world(
    const std::array<double, 3>& position_world_m) const {
  const double position_world_km[3] = {
      position_world_m[0] * 1.0e-3,
      position_world_m[1] * 1.0e-3,
      position_world_m[2] * 1.0e-3};
  double lvlh_km[3];
  detail::mat3_mul_vec(orbit_instance_->C_LI, position_world_km, lvlh_km);
  return {1000.0 * lvlh_km[0], 1000.0 * lvlh_km[1], 1000.0 * lvlh_km[2]};
}

std::array<double, 3> MjoData::lvlh_velocity_from_world(
    const std::array<double, 3>& position_world_m,
    const std::array<double, 3>& velocity_world_m_s) const {
  const auto position_lvlh_m = lvlh_position_from_world(position_world_m);
  const double position_lvlh_km[3] = {
      position_lvlh_m[0] * 1.0e-3,
      position_lvlh_m[1] * 1.0e-3,
      position_lvlh_m[2] * 1.0e-3};
  const double velocity_world_km_s[3] = {
      velocity_world_m_s[0] * 1.0e-3,
      velocity_world_m_s[1] * 1.0e-3,
      velocity_world_m_s[2] * 1.0e-3};
  double lvlh_rot[3];
  detail::mat3_mul_vec(orbit_instance_->C_LI, velocity_world_km_s, lvlh_rot);
  double omega_cross_r[3];
  detail::cross3(orbit_instance_->omega_lvlh, position_lvlh_km, omega_cross_r);
  double lvlh_km_s[3];
  detail::sub3(lvlh_rot, omega_cross_r, lvlh_km_s);
  return {1000.0 * lvlh_km_s[0], 1000.0 * lvlh_km_s[1], 1000.0 * lvlh_km_s[2]};
}

}  // namespace mujoco_orbit
