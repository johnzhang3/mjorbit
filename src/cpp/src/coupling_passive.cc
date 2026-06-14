#include "mjorbit/coupling.h"

#include <algorithm>
#include <cmath>

#include <mujoco/mujoco.h>

#include "mjorbit/environment.h"
#include "mjorbit/gravity.h"
#include "mjorbit/math_utils.h"

namespace mjorbit {
namespace {

constexpr double kMToKm = 1.0e-3;
constexpr double kKmS2ToMS2 = 1.0e3;

void body_eci_position_km(
    const OrbitInstance* inst,
    const mjData* d,
    int body_id,
    const double offset_world_m[3],
    double out_r_eci[3]) {
  const mjtNum* xpos = d->xipos + 3 * body_id;
  out_r_eci[0] = inst->R_eci[0] + (static_cast<double>(xpos[0]) + offset_world_m[0]) * kMToKm;
  out_r_eci[1] = inst->R_eci[1] + (static_cast<double>(xpos[1]) + offset_world_m[1]) * kMToKm;
  out_r_eci[2] = inst->R_eci[2] + (static_cast<double>(xpos[2]) + offset_world_m[2]) * kMToKm;
}

void body_eci_velocity_km_s(
    const OrbitInstance* inst,
    const double velocity_world_m_s[3],
    double out_v_eci[3]) {
  out_v_eci[0] = inst->V_eci[0] + velocity_world_m_s[0] * kMToKm;
  out_v_eci[1] = inst->V_eci[1] + velocity_world_m_s[1] * kMToKm;
  out_v_eci[2] = inst->V_eci[2] + velocity_world_m_s[2] * kMToKm;
}

void add_force_torque_at_com(
    const mjModel* m,
    mjData* d,
    OrbitInstance* inst,
    int body_id,
    const double force_world[3],
    const double torque_world[3]) {
  // Per-body force/torque accumulator. apply_passive_wrenches makes one
  // mj_applyFT call per body after all sources have contributed, which
  // amortises the body Jacobian cost across all sources hitting that body.
  if (inst->wrench_buffer && body_id >= 0 && body_id < inst->wrench_body_count) {
    double* wrench = inst->wrench_buffer + 6 * body_id;
    for (int i = 0; i < 3; ++i) {
      wrench[i] += force_world[i];
      wrench[3 + i] += torque_world[i];
    }
    return;
  }

  if (!m || !d || body_id < 0 || body_id >= m->nbody) {
    return;
  }
  const mjtNum point[3] = {
      d->xipos[3 * body_id + 0],
      d->xipos[3 * body_id + 1],
      d->xipos[3 * body_id + 2],
  };
  const mjtNum force[3] = {force_world[0], force_world[1], force_world[2]};
  const mjtNum torque[3] = {torque_world[0], torque_world[1], torque_world[2]};
  mj_applyFT(m, d, force, torque, point, body_id, d->qfrc_passive);
}

void flush_wrenches_to_qfrc(const mjModel* m, mjData* d, const OrbitInstance* inst) {
  if (!inst->wrench_buffer) {
    return;
  }
  const int nbody = std::min(static_cast<int>(m->nbody), inst->wrench_body_count);
  for (int body_id = 1; body_id < nbody; ++body_id) {
    const double* wrench = inst->wrench_buffer + 6 * body_id;
    bool any_nonzero = false;
    for (int i = 0; i < 6; ++i) {
      if (wrench[i] != 0.0) {
        any_nonzero = true;
        break;
      }
    }
    if (!any_nonzero) {
      continue;
    }
    const mjtNum point[3] = {
        d->xipos[3 * body_id + 0],
        d->xipos[3 * body_id + 1],
        d->xipos[3 * body_id + 2],
    };
    const mjtNum force[3] = {
        static_cast<mjtNum>(wrench[0]),
        static_cast<mjtNum>(wrench[1]),
        static_cast<mjtNum>(wrench[2]),
    };
    const mjtNum torque[3] = {
        static_cast<mjtNum>(wrench[3]),
        static_cast<mjtNum>(wrench[4]),
        static_cast<mjtNum>(wrench[5]),
    };
    mj_applyFT(m, d, force, torque, point, body_id, d->qfrc_passive);
  }
}

void add_feedback_force(OrbitInstance* inst, const double force_world[3]) {
  for (int i = 0; i < 3; ++i) {
    inst->feedback_force_world[i] += force_world[i];
  }
}

double total_body_mass(const mjModel* m) {
  double total = 0.0;
  for (int body_id = 1; body_id < m->nbody; ++body_id) {
    total += std::max(0.0, static_cast<double>(m->body_mass[body_id]));
  }
  return total;
}

void xmat_to_double(const mjData* d, int body_id, double out_R_body[9]) {
  const mjtNum* xmat = d->xmat + 9 * body_id;
  for (int i = 0; i < 9; ++i) {
    out_R_body[i] = static_cast<double>(xmat[i]);
  }
}

void world_to_body(const double R_body[9], const double world[3], double out_body[3]) {
  out_body[0] = R_body[0] * world[0] + R_body[3] * world[1] + R_body[6] * world[2];
  out_body[1] = R_body[1] * world[0] + R_body[4] * world[1] + R_body[7] * world[2];
  out_body[2] = R_body[2] * world[0] + R_body[5] * world[1] + R_body[8] * world[2];
}

void body_angular_velocity_world(const mjData* d, int body_id, double out_w_world[3]) {
  const mjtNum* cvel = d->cvel + 6 * body_id;
  out_w_world[0] = static_cast<double>(cvel[0]);
  out_w_world[1] = static_cast<double>(cvel[1]);
  out_w_world[2] = static_cast<double>(cvel[2]);
}

void apply_inertial_wrenches(const mjModel* m, mjData* d, OrbitInstance* inst) {
  for (int body_id = 1; body_id < m->nbody; ++body_id) {
    const double mass = m->body_mass[body_id];
    if (mass <= 0.0) {
      continue;
    }

    const double zero[3] = {0.0, 0.0, 0.0};
    const mjtNum* xpos = d->xipos + 3 * body_id;
    const double rho_km[3] = {
        static_cast<double>(xpos[0]) * kMToKm,
        static_cast<double>(xpos[1]) * kMToKm,
        static_cast<double>(xpos[2]) * kMToKm,
    };

    // Encke's identity (paper eq:encke) for the differential gravity, which
    // is well-conditioned at single precision unlike a literal subtraction
    // of the two large gravity vectors.
    double diff_accel[3];
    relative_accel(
        rho_km, inst->R_eci, diff_accel, inst->use_j2 != 0, inst->central_body);

    double diff_force[3];
    for (int i = 0; i < 3; ++i) {
      diff_force[i] = mass * diff_accel[i] * kKmS2ToMS2;
    }
    add_force_torque_at_com(m, d, inst, body_id, diff_force, zero);
  }
}

void apply_gravity_gradient_torques(const mjModel* m, mjData* d, OrbitInstance* inst) {
  if (!inst->use_gravity_gradient) {
    return;
  }

  for (int body_id = 1; body_id < m->nbody; ++body_id) {
    if (m->body_mass[body_id] <= 0.0) {
      continue;
    }

    const double zero[3] = {0.0, 0.0, 0.0};
    double r_body_eci[3];
    body_eci_position_km(inst, d, body_id, zero, r_body_eci);
    const double r_mag = detail::norm3(r_body_eci);
    if (r_mag < 1.0e-9) {
      continue;
    }

    double r_hat[3];
    detail::scale3(r_body_eci, 1.0 / r_mag, r_hat);

    const mjtNum* ximat = d->ximat + 9 * body_id;
    const mjtNum* inertia = m->body_inertia + 3 * body_id;

    double J_world[9];
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        J_world[3 * row + col] =
            static_cast<double>(ximat[3 * row + 0]) * static_cast<double>(inertia[0]) *
                static_cast<double>(ximat[3 * col + 0]) +
            static_cast<double>(ximat[3 * row + 1]) * static_cast<double>(inertia[1]) *
                static_cast<double>(ximat[3 * col + 1]) +
            static_cast<double>(ximat[3 * row + 2]) * static_cast<double>(inertia[2]) *
                static_cast<double>(ximat[3 * col + 2]);
      }
    }

    double J_rhat[3];
    detail::mat3_mul_vec(J_world, r_hat, J_rhat);

    double torque_world[3];
    detail::cross3(r_hat, J_rhat, torque_world);
    const double coeff = 3.0 * inst->central_body.gm / std::pow(r_mag, 3);
    detail::scale3(torque_world, coeff, torque_world);

    add_force_torque_at_com(m, d, inst, body_id, zero, torque_world);
  }
}

void apply_magnetic_wrenches(const mjModel* m, mjData* d, OrbitInstance* inst) {
  if (!inst->use_magnetic || inst->num_magnetic_bodies <= 0 || !inst->magnetic_bodies) {
    return;
  }

  const double zero[3] = {0.0, 0.0, 0.0};
  for (int idx = 0; idx < inst->num_magnetic_bodies; ++idx) {
    const MagneticMetadataNative& magnetic = inst->magnetic_bodies[idx];
    const int body_id = magnetic.body_id;
    if (body_id <= 0 || body_id >= m->nbody) {
      continue;
    }

    double R_body[9];
    xmat_to_double(d, body_id, R_body);
    double B_body[3];
    world_to_body(R_body, inst->mag_field_eci, B_body);

    double tau_body[3];
    detail::cross3(magnetic.dipole_body, B_body, tau_body);

    double tau_world[3];
    detail::mat3_mul_vec(R_body, tau_body, tau_world);
    add_force_torque_at_com(m, d, inst, body_id, zero, tau_world);
  }
}

void apply_surface_wrenches(const mjModel* m, mjData* d, OrbitInstance* inst) {
  if (inst->num_surfaces <= 0 || !inst->surfaces) {
    return;
  }

  for (int idx = 0; idx < inst->num_surfaces; ++idx) {
    const SurfaceMetadataNative& surface = inst->surfaces[idx];
    const int body_id = surface.body_id;
    if (body_id <= 0 || body_id >= m->nbody) {
      continue;
    }

    if ((!surface.use_drag || !inst->use_drag) && (!surface.use_srp || !inst->use_srp)) {
      continue;
    }

    const mjtNum* cvel = d->cvel + 6 * body_id;

    double R_body[9];
    xmat_to_double(d, body_id, R_body);

    double r_cop_world[3];
    detail::mat3_mul_vec(R_body, surface.center_of_pressure_body, r_cop_world);
    double n_world[3];
    detail::mat3_mul_vec(R_body, surface.normal_body, n_world);

    double omega_body[3] = {
        static_cast<double>(cvel[0]),
        static_cast<double>(cvel[1]),
        static_cast<double>(cvel[2]),
    };
    double v_com[3] = {
        static_cast<double>(cvel[3]),
        static_cast<double>(cvel[4]),
        static_cast<double>(cvel[5]),
    };
    double omega_cross_r[3];
    detail::cross3(omega_body, r_cop_world, omega_cross_r);
    double v_point_world[3];
    detail::add3(v_com, omega_cross_r, v_point_world);

    double r_point_eci_km[3];
    body_eci_position_km(inst, d, body_id, r_cop_world, r_point_eci_km);
    double v_point_eci_km_s[3];
    body_eci_velocity_km_s(inst, v_point_world, v_point_eci_km_s);
    double v_point_eci_m_s[3];
    detail::scale3(v_point_eci_km_s, 1.0e3, v_point_eci_m_s);

    double v_atm_eci_km_s[3];
    detail::cross3(inst->atmosphere_omega_eci, r_point_eci_km, v_atm_eci_km_s);
    double v_atm_eci_m_s[3];
    detail::scale3(v_atm_eci_km_s, 1.0e3, v_atm_eci_m_s);

    double v_rel_m_s[3];
    detail::sub3(v_point_eci_m_s, v_atm_eci_m_s, v_rel_m_s);
    const double speed = detail::norm3(v_rel_m_s);

    double total_force[3] = {0.0, 0.0, 0.0};

    if (surface.use_drag && inst->use_drag && speed > 1.0e-10) {
      double v_hat[3];
      detail::scale3(v_rel_m_s, 1.0 / speed, v_hat);
      const double cos_angle = detail::dot3(n_world, v_hat);
      if (cos_angle > 0.0) {
        // Evaluate density at this surface's own ECI position, not the chief's
        // cached scalar, so bodies offset in altitude see the correct drag.
        // Matches the Warp kernel and the Python reference.
        const double rho_local = atm_density(r_point_eci_km, inst->central_body);
        const double projected_area = surface.area * cos_angle;
        const double drag_scale = -0.5 * rho_local * surface.drag_coeff *
                                  projected_area * speed * speed;
        double drag_force[3];
        detail::scale3(v_hat, drag_scale, drag_force);
        detail::add3(total_force, drag_force, total_force);
      }
    }

    if (surface.use_srp && inst->use_srp) {
      const double cos_sun = detail::dot3(n_world, inst->sun_vector_eci);
      if (cos_sun > 0.0) {
        // Evaluate the shadow at this surface's own ECI position, not the
        // chief's cached scalar, so a formation straddling the terminator gets
        // the correct per-surface SRP. Matches the Warp kernel and reference.
        const double eclipse_local =
            eclipse_factor(r_point_eci_km, inst->sun_vector_eci);
        if (eclipse_local > 0.0) {
          const double projected_area = surface.area * cos_sun;
          const double srp_scale =
              -eclipse_local * kPSun * surface.srp_coeff * projected_area;
          double srp_force[3];
          detail::scale3(inst->sun_vector_eci, srp_scale, srp_force);
          detail::add3(total_force, srp_force, total_force);
        }
      }
    }

    if (detail::norm3(total_force) == 0.0) {
      continue;
    }

    double torque_world[3];
    detail::cross3(r_cop_world, total_force, torque_world);
    add_force_torque_at_com(m, d, inst, body_id, total_force, torque_world);
    add_feedback_force(inst, total_force);
  }
}

double effective_rw_alpha(
    const ReactionWheelMetadataNative& rw,
    double speed,
    double torque_cmd) {
  if (rw.inertia <= 0.0) {
    return 0.0;
  }
  double tau = torque_cmd;
  if (rw.has_torque_limit) {
    tau = detail::clamp(tau, -rw.torque_limit, rw.torque_limit);
  }
  double alpha = tau / rw.inertia;
  if (rw.has_speed_limit) {
    if (speed >= rw.speed_limit && alpha > 0.0) {
      alpha = 0.0;
    } else if (speed <= -rw.speed_limit && alpha < 0.0) {
      alpha = 0.0;
    }
  }
  return alpha;
}

void apply_reaction_wheel_wrenches(const mjModel* m, mjData* d, OrbitInstance* inst) {
  if (inst->num_reaction_wheels <= 0 || !inst->reaction_wheels) {
    return;
  }

  const double zero[3] = {0.0, 0.0, 0.0};
  for (int idx = 0; idx < inst->num_reaction_wheels; ++idx) {
    const ReactionWheelMetadataNative& rw = inst->reaction_wheels[idx];
    const int body_id = rw.body_id;
    if (body_id <= 0 || body_id >= m->nbody) {
      continue;
    }

    const double speed = inst->rw_speed ? inst->rw_speed[idx] : 0.0;
    const double inertia = rw.inertia;
    if (inst->rw_momentum) {
      inst->rw_momentum[idx] = speed * inertia;
    }
    double R_body[9];
    xmat_to_double(d, body_id, R_body);
    double w_world[3];
    body_angular_velocity_world(d, body_id, w_world);
    double w_body[3];
    world_to_body(R_body, w_world, w_body);

    double h_body[3];
    detail::scale3(rw.axis_body, inertia * speed, h_body);
    double gyro_tau_body[3];
    detail::cross3(w_body, h_body, gyro_tau_body);
    detail::scale3(gyro_tau_body, -1.0, gyro_tau_body);

    double cmd_tau_body[3] = {0.0, 0.0, 0.0};
    if (inst->rw_torque_cmd) {
      const double alpha = effective_rw_alpha(rw, speed, inst->rw_torque_cmd[idx]);
      detail::scale3(rw.axis_body, -inertia * alpha, cmd_tau_body);
    }

    double tau_body[3];
    detail::add3(gyro_tau_body, cmd_tau_body, tau_body);
    double tau_world[3];
    detail::mat3_mul_vec(R_body, tau_body, tau_world);
    add_force_torque_at_com(m, d, inst, body_id, zero, tau_world);
  }
}

void cmg_momentum_body(
    double rotor_momentum,
    double gimbal_angle,
    const double spin_axis_0[3],
    const double torque_axis_0[3],
    double out_h_body[3],
    double out_torque_axis_body[3]) {
  const double c = std::cos(gimbal_angle);
  const double s = std::sin(gimbal_angle);
  for (int i = 0; i < 3; ++i) {
    out_h_body[i] = rotor_momentum * (c * spin_axis_0[i] + s * torque_axis_0[i]);
    out_torque_axis_body[i] = c * torque_axis_0[i] - s * spin_axis_0[i];
  }
}

double effective_cmg_rate(
    const ControlMomentGyroMetadataNative& cmg,
    double gimbal_angle,
    double rate_cmd,
    double dt,
    double* out_new_angle = nullptr) {
  double theta_dot = rate_cmd;
  if (cmg.has_gimbal_rate_limit) {
    theta_dot = detail::clamp(theta_dot, -cmg.gimbal_rate_limit, cmg.gimbal_rate_limit);
  }
  if (cmg.has_gimbal_angle_limit) {
    const double limit = cmg.gimbal_angle_limit;
    if (gimbal_angle >= limit && theta_dot > 0.0) {
      theta_dot = 0.0;
    } else if (gimbal_angle <= -limit && theta_dot < 0.0) {
      theta_dot = 0.0;
    }
  }

  double theta_new = gimbal_angle + theta_dot * dt;
  if (cmg.has_gimbal_angle_limit) {
    theta_new = detail::clamp(theta_new, -cmg.gimbal_angle_limit, cmg.gimbal_angle_limit);
  }
  if (out_new_angle) {
    *out_new_angle = theta_new;
  }
  return dt > 0.0 ? (theta_new - gimbal_angle) / dt : 0.0;
}

void apply_cmg_wrenches(const mjModel* m, mjData* d, OrbitInstance* inst) {
  if (inst->num_cmgs <= 0 || !inst->cmgs) {
    return;
  }

  const double zero[3] = {0.0, 0.0, 0.0};
  const double dt = m->opt.timestep;
  for (int idx = 0; idx < inst->num_cmgs; ++idx) {
    const ControlMomentGyroMetadataNative& cmg = inst->cmgs[idx];
    const int body_id = cmg.body_id;
    if (body_id <= 0 || body_id >= m->nbody) {
      continue;
    }

    const double theta = inst->cmg_gimbal_angle ? inst->cmg_gimbal_angle[idx] : 0.0;
    const double h_mag = inst->cmg_rotor_momentum ? inst->cmg_rotor_momentum[idx] : cmg.rotor_momentum;
    if (h_mag <= 0.0) {
      continue;
    }

    double h_body[3];
    double torque_axis_body[3];
    cmg_momentum_body(
        h_mag,
        theta,
        cmg.spin_axis_body_0,
        cmg.torque_axis_body_0,
        h_body,
        torque_axis_body);

    double R_body[9];
    xmat_to_double(d, body_id, R_body);
    double w_world[3];
    body_angular_velocity_world(d, body_id, w_world);
    double w_body[3];
    world_to_body(R_body, w_world, w_body);

    double gyro_tau_body[3];
    detail::cross3(w_body, h_body, gyro_tau_body);
    detail::scale3(gyro_tau_body, -1.0, gyro_tau_body);

    double cmd_tau_body[3] = {0.0, 0.0, 0.0};
    if (inst->cmg_gimbal_rate_cmd) {
      const double effective_rate =
          effective_cmg_rate(cmg, theta, inst->cmg_gimbal_rate_cmd[idx], dt);
      detail::scale3(torque_axis_body, -h_mag * effective_rate, cmd_tau_body);
    }

    double tau_body[3];
    detail::add3(gyro_tau_body, cmd_tau_body, tau_body);
    double tau_world[3];
    detail::mat3_mul_vec(R_body, tau_body, tau_world);
    add_force_torque_at_com(m, d, inst, body_id, zero, tau_world);
  }
}

void apply_magnetorquer_wrenches(const mjModel* m, mjData* d, OrbitInstance* inst) {
  if (!inst->use_magnetic || inst->num_magnetorquers <= 0 || !inst->magnetorquers ||
      !inst->mtq_dipole_cmd) {
    return;
  }

  const double zero[3] = {0.0, 0.0, 0.0};
  for (int idx = 0; idx < inst->num_magnetorquers; ++idx) {
    const MagnetorquerMetadataNative& mtq = inst->magnetorquers[idx];
    const int body_id = mtq.body_id;
    if (body_id <= 0 || body_id >= m->nbody) {
      continue;
    }

    const double m_cmd = detail::clamp(
        inst->mtq_dipole_cmd[idx],
        -mtq.dipole_limit,
        mtq.dipole_limit);
    double dipole_body[3];
    detail::scale3(mtq.axis_body, m_cmd, dipole_body);

    double R_body[9];
    xmat_to_double(d, body_id, R_body);
    double B_body[3];
    world_to_body(R_body, inst->mag_field_eci, B_body);

    double tau_body[3];
    detail::cross3(dipole_body, B_body, tau_body);
    double tau_world[3];
    detail::mat3_mul_vec(R_body, tau_body, tau_world);
    add_force_torque_at_com(m, d, inst, body_id, zero, tau_world);
  }
}

void apply_thruster_wrenches(const mjModel* m, mjData* d, OrbitInstance* inst) {
  if (inst->num_thrusters <= 0 || !inst->thrusters || !inst->thr_force_cmd) {
    return;
  }

  for (int idx = 0; idx < inst->num_thrusters; ++idx) {
    const ThrusterMetadataNative& thr = inst->thrusters[idx];
    const int body_id = thr.body_id;
    if (body_id <= 0 || body_id >= m->nbody) {
      continue;
    }

    const double f_cmd = detail::clamp(inst->thr_force_cmd[idx], 0.0, thr.force_limit);
    if (f_cmd == 0.0) {
      continue;
    }

    double R_body[9];
    xmat_to_double(d, body_id, R_body);
    double force_body[3];
    detail::scale3(thr.direction_body, f_cmd, force_body);
    double force_world[3];
    detail::mat3_mul_vec(R_body, force_body, force_world);

    const mjtNum* ipos = m->body_ipos + 3 * body_id;
    double r_com_body[3] = {
        thr.position_body[0] - static_cast<double>(ipos[0]),
        thr.position_body[1] - static_cast<double>(ipos[1]),
        thr.position_body[2] - static_cast<double>(ipos[2]),
    };
    double tau_body[3];
    detail::cross3(r_com_body, force_body, tau_body);
    double tau_world[3];
    detail::mat3_mul_vec(R_body, tau_body, tau_world);

    add_force_torque_at_com(m, d, inst, body_id, force_world, tau_world);
    add_feedback_force(inst, force_world);
  }
}

void compute_feedback_accel(const mjModel* m, OrbitInstance* inst) {
  const double mass = total_body_mass(m);
  if (mass <= 0.0) {
    detail::zero3(inst->feedback_accel_eci);
    return;
  }
  for (int i = 0; i < 3; ++i) {
    inst->feedback_accel_eci[i] = (inst->feedback_force_world[i] / mass) * kMToKm;
  }
}

void apply_origin_acceleration_wrenches(const mjModel* m, mjData* d, OrbitInstance* inst) {
  double accel_m_s2[3];
  detail::scale3(inst->feedback_accel_eci, kKmS2ToMS2, accel_m_s2);
  if (detail::norm3(accel_m_s2) == 0.0) {
    return;
  }

  const double zero_torque[3] = {0.0, 0.0, 0.0};
  for (int body_id = 1; body_id < m->nbody; ++body_id) {
    const double mass = m->body_mass[body_id];
    if (mass <= 0.0) {
      continue;
    }
    double force_world[3];
    detail::scale3(accel_m_s2, -mass, force_world);
    add_force_torque_at_com(m, d, inst, body_id, force_world, zero_torque);
  }
}

void advance_reaction_wheels(const mjModel* m, OrbitInstance* inst) {
  if (inst->num_reaction_wheels <= 0 || !inst->reaction_wheels || !inst->rw_speed) {
    return;
  }
  const double dt = m->opt.timestep;
  for (int idx = 0; idx < inst->num_reaction_wheels; ++idx) {
    const ReactionWheelMetadataNative& rw = inst->reaction_wheels[idx];
    const double torque_cmd = inst->rw_torque_cmd ? inst->rw_torque_cmd[idx] : 0.0;
    const double alpha = effective_rw_alpha(rw, inst->rw_speed[idx], torque_cmd);
    inst->rw_speed[idx] += alpha * dt;
    if (rw.has_speed_limit) {
      inst->rw_speed[idx] = detail::clamp(inst->rw_speed[idx], -rw.speed_limit, rw.speed_limit);
    }
    if (inst->rw_momentum) {
      inst->rw_momentum[idx] = inst->rw_speed[idx] * rw.inertia;
    }
  }
}

void advance_cmgs(const mjModel* m, OrbitInstance* inst) {
  if (inst->num_cmgs <= 0 || !inst->cmgs || !inst->cmg_gimbal_angle) {
    return;
  }
  const double dt = m->opt.timestep;
  for (int idx = 0; idx < inst->num_cmgs; ++idx) {
    const double rate_cmd = inst->cmg_gimbal_rate_cmd ? inst->cmg_gimbal_rate_cmd[idx] : 0.0;
    double theta_new = inst->cmg_gimbal_angle[idx];
    effective_cmg_rate(inst->cmgs[idx], inst->cmg_gimbal_angle[idx], rate_cmd, dt, &theta_new);
    inst->cmg_gimbal_angle[idx] = theta_new;
  }
}

}  // namespace

void apply_passive_wrenches(const mjModel* m, mjData* d, OrbitInstance* inst) {
  if (!m || !d || !inst) {
    return;
  }

  if (inst->wrench_buffer) {
    const int nbody = std::min(static_cast<int>(m->nbody), inst->wrench_body_count);
    std::fill(inst->wrench_buffer, inst->wrench_buffer + 6 * nbody, 0.0);
  }
  detail::zero3(inst->feedback_force_world);
  detail::zero3(inst->feedback_accel_eci);

  apply_inertial_wrenches(m, d, inst);
  apply_surface_wrenches(m, d, inst);
  apply_magnetic_wrenches(m, d, inst);
  apply_gravity_gradient_torques(m, d, inst);
  apply_reaction_wheel_wrenches(m, d, inst);
  apply_cmg_wrenches(m, d, inst);
  apply_magnetorquer_wrenches(m, d, inst);
  apply_thruster_wrenches(m, d, inst);

  compute_feedback_accel(m, inst);
  apply_origin_acceleration_wrenches(m, d, inst);

  flush_wrenches_to_qfrc(m, d, inst);
}

void advance_actuators(const mjModel* m, OrbitInstance* inst) {
  if (!m || !inst) {
    return;
  }
  advance_reaction_wheels(m, inst);
  advance_cmgs(m, inst);
}

}  // namespace mjorbit
