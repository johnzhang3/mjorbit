#include "mjorbit/lvlh.h"

#include "mjorbit/gravity.h"
#include "mjorbit/math_utils.h"

namespace mjorbit {

void update_frame_cache(
    const double R_eci[3],
    const double V_eci[3],
    FrameCache* out_cache,
    bool use_j2) {
  update_frame_cache(R_eci, V_eci, out_cache, use_j2, CentralBodySpecNative{});
}

void update_frame_cache(
    const double R_eci[3],
    const double V_eci[3],
    FrameCache* out_cache,
    bool use_j2,
    const CentralBodySpecNative& central_body) {
  if (!out_cache) {
    return;
  }

  const double r = detail::norm3(R_eci);
  double h[3];
  detail::cross3(R_eci, V_eci, h);
  const double h_mag = detail::norm3(h);
  if (r == 0.0 || h_mag == 0.0) {
    detail::set_identity3(out_cache->C_LI);
    detail::set_identity3(out_cache->C_IL);
    detail::zero3(out_cache->omega_lvlh);
    detail::zero3(out_cache->omega_dot_lvlh);
    return;
  }

  double x_hat[3];
  detail::normalize3(R_eci, x_hat);
  double z_hat[3];
  detail::normalize3(h, z_hat);
  double y_hat[3];
  detail::cross3(z_hat, x_hat, y_hat);

  out_cache->C_LI[0] = x_hat[0];
  out_cache->C_LI[1] = x_hat[1];
  out_cache->C_LI[2] = x_hat[2];
  out_cache->C_LI[3] = y_hat[0];
  out_cache->C_LI[4] = y_hat[1];
  out_cache->C_LI[5] = y_hat[2];
  out_cache->C_LI[6] = z_hat[0];
  out_cache->C_LI[7] = z_hat[1];
  out_cache->C_LI[8] = z_hat[2];
  detail::mat3_transpose(out_cache->C_LI, out_cache->C_IL);

  double omega_eci[3];
  detail::scale3(h, 1.0 / (r * r), omega_eci);
  detail::mat3_mul_vec(out_cache->C_LI, omega_eci, out_cache->omega_lvlh);

  double a_eci[3];
  total_accel(R_eci, a_eci, use_j2, central_body);
  double dh_dt[3];
  detail::cross3(R_eci, a_eci, dh_dt);
  const double dr_dt = detail::dot3(R_eci, V_eci) / r;
  double domega_dt_eci[3];
  for (int i = 0; i < 3; ++i) {
    domega_dt_eci[i] = dh_dt[i] / (r * r) - 2.0 * h[i] * dr_dt / (r * r * r);
  }
  detail::mat3_mul_vec(out_cache->C_LI, domega_dt_eci, out_cache->omega_dot_lvlh);
}

void eci_to_lvlh_pos(const double r_eci[3], const double R_ref[3], const double C_LI[9], double out_r_lvlh[3]) {
  double delta[3];
  detail::sub3(r_eci, R_ref, delta);
  detail::mat3_mul_vec(C_LI, delta, out_r_lvlh);
}

void lvlh_to_eci_pos(const double r_lvlh[3], const double R_ref[3], const double C_IL[9], double out_r_eci[3]) {
  double delta_eci[3];
  detail::mat3_mul_vec(C_IL, r_lvlh, delta_eci);
  detail::add3(R_ref, delta_eci, out_r_eci);
}

void eci_to_lvlh_vel(
    const double v_eci[3],
    const double r_lvlh[3],
    const double V_ref[3],
    const double C_LI[9],
    const double omega_lvlh[3],
    double out_v_lvlh[3]) {
  double delta_v[3];
  detail::sub3(v_eci, V_ref, delta_v);
  double delta_v_lvlh[3];
  detail::mat3_mul_vec(C_LI, delta_v, delta_v_lvlh);
  double omega_cross_r[3];
  detail::cross3(omega_lvlh, r_lvlh, omega_cross_r);
  detail::sub3(delta_v_lvlh, omega_cross_r, out_v_lvlh);
}

}  // namespace mjorbit
