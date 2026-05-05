#include "mujoco_orbit/gravity.h"

#include "mujoco_orbit/math_utils.h"

namespace mujoco_orbit {

void point_mass_accel(const double r_eci[3], double out_a[3], double gm) {
  const double r = detail::norm3(r_eci);
  if (r == 0.0) {
    detail::zero3(out_a);
    return;
  }
  const double factor = -gm / (r * r * r);
  detail::scale3(r_eci, factor, out_a);
}

void j2_accel(const double r_eci[3], double out_a[3], double gm, double j2, double r_eq) {
  const double x = r_eci[0];
  const double y = r_eci[1];
  const double z = r_eci[2];
  const double r = detail::norm3(r_eci);
  if (r == 0.0) {
    detail::zero3(out_a);
    return;
  }
  const double factor = 1.5 * j2 * gm * r_eq * r_eq / std::pow(r, 5);
  const double z_r2 = (z / r) * (z / r);
  out_a[0] = factor * x * (5.0 * z_r2 - 1.0);
  out_a[1] = factor * y * (5.0 * z_r2 - 1.0);
  out_a[2] = factor * z * (5.0 * z_r2 - 3.0);
}

void total_accel(const double r_eci[3], double out_a[3], bool use_j2, double gm, double j2, double r_eq) {
  point_mass_accel(r_eci, out_a, gm);
  if (!use_j2) {
    return;
  }
  double a_j2[3];
  j2_accel(r_eci, a_j2, gm, j2, r_eq);
  detail::add3(out_a, a_j2, out_a);
}

void total_accel(
    const double r_eci[3],
    double out_a[3],
    bool use_j2,
    const CentralBodySpecNative& central_body) {
  total_accel(
      r_eci,
      out_a,
      use_j2,
      central_body.gm,
      central_body.j2,
      central_body.radius);
}

void encke_point_mass_relative_accel(
    const double rho[3],
    const double r_chief[3],
    double out_a[3],
    double gm) {
  const double rc2 = detail::dot3(r_chief, r_chief);
  if (rc2 == 0.0) {
    detail::zero3(out_a);
    return;
  }
  const double rc = std::sqrt(rc2);

  // sigma = rho . (2 r_chief + rho) / r_chief^2, computed without forming
  // 2 r_chief + rho explicitly so we can keep it stable for tiny rho.
  const double sigma =
      (2.0 * detail::dot3(rho, r_chief) + detail::dot3(rho, rho)) / rc2;

  // f(sigma) = 1 - (1+sigma)^{-3/2}, evaluated as
  // sigma * (3 + 3 sigma + sigma^2) / [(1 + (1+sigma)^{3/2}) * (1+sigma)^{3/2}]
  // to avoid cancellation when sigma -> 0.
  const double one_plus_sigma = 1.0 + sigma;
  const double one_plus_sigma_3_2 = one_plus_sigma * std::sqrt(one_plus_sigma);
  const double f =
      sigma * (3.0 + 3.0 * sigma + sigma * sigma) /
      ((1.0 + one_plus_sigma_3_2) * one_plus_sigma_3_2);

  const double scale = -gm / (rc2 * rc);
  for (int i = 0; i < 3; ++i) {
    out_a[i] = scale * (rho[i] - f * (r_chief[i] + rho[i]));
  }
}

void relative_accel(
    const double rho[3],
    const double r_chief[3],
    double out_a[3],
    bool use_j2,
    const CentralBodySpecNative& central_body) {
  encke_point_mass_relative_accel(rho, r_chief, out_a, central_body.gm);
  if (!use_j2) {
    return;
  }
  // J2 differential is much smaller than the point-mass term, so direct
  // subtraction is numerically fine.
  double r_body[3];
  detail::add3(r_chief, rho, r_body);
  double j2_body[3];
  double j2_chief[3];
  j2_accel(r_body, j2_body, central_body.gm, central_body.j2, central_body.radius);
  j2_accel(r_chief, j2_chief, central_body.gm, central_body.j2, central_body.radius);
  for (int i = 0; i < 3; ++i) {
    out_a[i] += j2_body[i] - j2_chief[i];
  }
}

}  // namespace mujoco_orbit
