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

}  // namespace mujoco_orbit
