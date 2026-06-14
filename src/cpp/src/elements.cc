#include "mjorbit/elements.h"

#include <cmath>

#include "mjorbit/math_utils.h"

namespace mjorbit {

void rot_pf_to_eci(double raan, double inc, double argp, double out_R[9]) {
  const double c_O = std::cos(raan);
  const double s_O = std::sin(raan);
  const double c_i = std::cos(inc);
  const double s_i = std::sin(inc);
  const double c_w = std::cos(argp);
  const double s_w = std::sin(argp);

  out_R[0] = c_O * c_w - s_O * s_w * c_i;
  out_R[1] = -c_O * s_w - s_O * c_w * c_i;
  out_R[2] = s_O * s_i;
  out_R[3] = s_O * c_w + c_O * s_w * c_i;
  out_R[4] = -s_O * s_w + c_O * c_w * c_i;
  out_R[5] = -c_O * s_i;
  out_R[6] = s_w * s_i;
  out_R[7] = c_w * s_i;
  out_R[8] = c_i;
}

void keplerian_to_cartesian(
    double a,
    double e,
    double inc,
    double raan,
    double argp,
    double nu,
    double out_R_eci[3],
    double out_V_eci[3],
    double gm) {
  const double p = a * (1.0 - e * e);
  const double r_pqw = p / (1.0 + e * std::cos(nu));
  const double cos_nu = std::cos(nu);
  const double sin_nu = std::sin(nu);

  const double R_pf[3] = {r_pqw * cos_nu, r_pqw * sin_nu, 0.0};
  const double sqrt_mu_over_p = std::sqrt(gm / p);
  const double V_pf[3] = {-sqrt_mu_over_p * sin_nu, sqrt_mu_over_p * (e + cos_nu), 0.0};

  double rot[9];
  rot_pf_to_eci(raan, inc, argp, rot);
  detail::mat3_mul_vec(rot, R_pf, out_R_eci);
  detail::mat3_mul_vec(rot, V_pf, out_V_eci);
}

void cartesian_to_keplerian(
    const double R_eci[3],
    const double V_eci[3],
    double* out_a,
    double* out_e,
    double* out_inc,
    double* out_raan,
    double* out_argp,
    double* out_nu,
    double gm) {
  const double r = detail::norm3(R_eci);
  const double v = detail::norm3(V_eci);

  double h[3];
  detail::cross3(R_eci, V_eci, h);
  const double h_mag = detail::norm3(h);

  const double k_hat[3] = {0.0, 0.0, 1.0};
  double n[3];
  detail::cross3(k_hat, h, n);
  const double n_mag = detail::norm3(n);

  const double rv_dot = detail::dot3(R_eci, V_eci);
  const double coeff = v * v - gm / r;
  double e_vec[3];
  for (int i = 0; i < 3; ++i) {
    e_vec[i] = (coeff * R_eci[i] - rv_dot * V_eci[i]) / gm;
  }
  const double e = detail::norm3(e_vec);

  const double energy = 0.5 * v * v - gm / r;
  const double a = -gm / (2.0 * energy);
  const double inc = std::acos(h[2] / h_mag);
  const double raan = (n_mag > 1e-12) ? std::atan2(n[1], n[0]) : 0.0;

  double argp = std::acos(detail::clamp(detail::dot3(n, e_vec) / (n_mag * e + 1e-30), -1.0, 1.0));
  if (e_vec[2] < 0.0) {
    argp = 2.0 * detail::kPi - argp;
  }

  double nu = std::acos(detail::clamp(detail::dot3(e_vec, R_eci) / (e * r + 1e-30), -1.0, 1.0));
  if (rv_dot < 0.0) {
    nu = 2.0 * detail::kPi - nu;
  }

  if (out_a) {
    *out_a = a;
  }
  if (out_e) {
    *out_e = e;
  }
  if (out_inc) {
    *out_inc = inc;
  }
  if (out_raan) {
    *out_raan = raan;
  }
  if (out_argp) {
    *out_argp = argp;
  }
  if (out_nu) {
    *out_nu = nu;
  }
}

}  // namespace mjorbit
