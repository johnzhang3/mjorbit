#include "mujoco_orbit/coupling.h"

#include <algorithm>
#include <cmath>

#include <mujoco/mujoco.h>

#include "mujoco_orbit/constants.h"
#include "mujoco_orbit/gravity.h"
#include "mujoco_orbit/math_utils.h"

namespace mujoco_orbit {
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
    int body_id,
    const double force_world[3],
    const double torque_world[3]) {
  const mjtNum point[3] = {
      d->xipos[3 * body_id + 0],
      d->xipos[3 * body_id + 1],
      d->xipos[3 * body_id + 2],
  };
  const mjtNum force[3] = {force_world[0], force_world[1], force_world[2]};
  const mjtNum torque[3] = {torque_world[0], torque_world[1], torque_world[2]};
  mj_applyFT(m, d, force, torque, point, body_id, d->qfrc_passive);
}

void apply_inertial_wrenches(const mjModel* m, mjData* d, const OrbitInstance* inst) {
  double chief_accel[3];
  total_accel(inst->R_eci, chief_accel, inst->use_j2 != 0);

  for (int body_id = 1; body_id < m->nbody; ++body_id) {
    const double mass = m->body_mass[body_id];
    if (mass <= 0.0) {
      continue;
    }

    const double zero[3] = {0.0, 0.0, 0.0};
    double r_body_eci[3];
    body_eci_position_km(inst, d, body_id, zero, r_body_eci);

    double g_body[3];
    total_accel(r_body_eci, g_body, inst->use_j2 != 0);

    double diff_force[3];
    for (int i = 0; i < 3; ++i) {
      diff_force[i] = mass * (g_body[i] - chief_accel[i]) * kKmS2ToMS2;
    }
    add_force_torque_at_com(m, d, body_id, diff_force, zero);
  }
}

void apply_gravity_gradient_torques(const mjModel* m, mjData* d, const OrbitInstance* inst) {
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
    const double coeff = 3.0 * kGmEarth / std::pow(r_mag, 3);
    detail::scale3(torque_world, coeff, torque_world);

    add_force_torque_at_com(m, d, body_id, zero, torque_world);
  }
}

void apply_magnetic_wrenches(const mjModel* m, mjData* d, const OrbitInstance* inst) {
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

    const mjtNum* xmat = d->xmat + 9 * body_id;
    double R_body[9];
    for (int i = 0; i < 9; ++i) {
      R_body[i] = static_cast<double>(xmat[i]);
    }
    double B_body[3] = {
        R_body[0] * inst->mag_field_eci[0] +
            R_body[3] * inst->mag_field_eci[1] +
            R_body[6] * inst->mag_field_eci[2],
        R_body[1] * inst->mag_field_eci[0] +
            R_body[4] * inst->mag_field_eci[1] +
            R_body[7] * inst->mag_field_eci[2],
        R_body[2] * inst->mag_field_eci[0] +
            R_body[5] * inst->mag_field_eci[1] +
            R_body[8] * inst->mag_field_eci[2],
    };

    double tau_body[3];
    detail::cross3(magnetic.dipole_body, B_body, tau_body);

    double tau_world[3];
    detail::mat3_mul_vec(R_body, tau_body, tau_world);
    add_force_torque_at_com(m, d, body_id, zero, tau_world);
  }
}

void apply_surface_wrenches(const mjModel* m, mjData* d, const OrbitInstance* inst) {
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

    const mjtNum* xmat = d->xmat + 9 * body_id;
    const mjtNum* cvel = d->cvel + 6 * body_id;

    double R_body[9];
    for (int i = 0; i < 9; ++i) {
      R_body[i] = static_cast<double>(xmat[i]);
    }

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
        const double projected_area = surface.area * cos_angle;
        const double drag_scale = -0.5 * inst->atm_density * surface.drag_coeff *
                                  projected_area * speed * speed;
        double drag_force[3];
        detail::scale3(v_hat, drag_scale, drag_force);
        detail::add3(total_force, drag_force, total_force);
      }
    }

    if (surface.use_srp && inst->use_srp && inst->eclipse > 0.0) {
      const double cos_sun = detail::dot3(n_world, inst->sun_vector_eci);
      if (cos_sun > 0.0) {
        const double projected_area = surface.area * cos_sun;
        const double srp_scale = -inst->eclipse * kPSun * surface.srp_coeff * projected_area;
        double srp_force[3];
        detail::scale3(inst->sun_vector_eci, srp_scale, srp_force);
        detail::add3(total_force, srp_force, total_force);
      }
    }

    if (detail::norm3(total_force) == 0.0) {
      continue;
    }

    double torque_world[3];
    detail::cross3(r_cop_world, total_force, torque_world);
    add_force_torque_at_com(m, d, body_id, total_force, torque_world);
  }
}

}  // namespace

void apply_passive_wrenches(const mjModel* m, mjData* d, const OrbitInstance* inst) {
  if (!m || !d || !inst) {
    return;
  }

  apply_inertial_wrenches(m, d, inst);
  apply_surface_wrenches(m, d, inst);
  apply_magnetic_wrenches(m, d, inst);
  apply_gravity_gradient_torques(m, d, inst);
}

}  // namespace mujoco_orbit
