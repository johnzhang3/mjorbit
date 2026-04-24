#include "mujoco_orbit/propagator.h"

#include "mujoco_orbit/gravity.h"
#include "mujoco_orbit/math_utils.h"

namespace mujoco_orbit {
namespace {

void derivatives(
    const double R[3],
    const double V[3],
    bool use_j2,
    const double* a_external,
    double out_dR[3],
    double out_dV[3]) {
  detail::copy3(V, out_dR);
  total_accel(R, out_dV, use_j2);
  if (a_external) {
    detail::add3(out_dV, a_external, out_dV);
  }
}

}  // namespace

void propagate_rk4(
    const double R_eci[3],
    const double V_eci[3],
    double t,
    double dt,
    double out_R_eci[3],
    double out_V_eci[3],
    double* out_t,
    bool use_j2,
    const double* a_external) {
  double k1R[3];
  double k1V[3];
  derivatives(R_eci, V_eci, use_j2, a_external, k1R, k1V);

  double R_tmp[3];
  double V_tmp[3];

  detail::add_scaled3(R_eci, k1R, 0.5 * dt, R_tmp);
  detail::add_scaled3(V_eci, k1V, 0.5 * dt, V_tmp);
  double k2R[3];
  double k2V[3];
  derivatives(R_tmp, V_tmp, use_j2, a_external, k2R, k2V);

  detail::add_scaled3(R_eci, k2R, 0.5 * dt, R_tmp);
  detail::add_scaled3(V_eci, k2V, 0.5 * dt, V_tmp);
  double k3R[3];
  double k3V[3];
  derivatives(R_tmp, V_tmp, use_j2, a_external, k3R, k3V);

  detail::add_scaled3(R_eci, k3R, dt, R_tmp);
  detail::add_scaled3(V_eci, k3V, dt, V_tmp);
  double k4R[3];
  double k4V[3];
  derivatives(R_tmp, V_tmp, use_j2, a_external, k4R, k4V);

  for (int i = 0; i < 3; ++i) {
    out_R_eci[i] = R_eci[i] + (dt / 6.0) * (k1R[i] + 2.0 * k2R[i] + 2.0 * k3R[i] + k4R[i]);
    out_V_eci[i] = V_eci[i] + (dt / 6.0) * (k1V[i] + 2.0 * k2V[i] + 2.0 * k3V[i] + k4V[i]);
  }

  if (out_t) {
    *out_t = t + dt;
  }
}

}  // namespace mujoco_orbit
