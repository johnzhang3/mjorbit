#include "mujoco_orbit/environment.h"

#include <cmath>

#include "mujoco_orbit/constants.h"
#include "mujoco_orbit/math_utils.h"

namespace mujoco_orbit {
namespace {

constexpr double kAtmH0Km = 400.0;
constexpr double kAtmRho0 = 2.62e-13;
constexpr double kAtmHScale = 58.2;

}  // namespace

void sun_vector_eci(double t, double out_sun_hat[3]) {
  const double T_jc = t / (36525.0 * 86400.0);
  const double lambda_sun = detail::deg2rad(280.460 + 36000.771 * T_jc);
  const double M_sun = detail::deg2rad(357.528 + 35999.050 * T_jc);
  const double lambda_ecl = lambda_sun + detail::deg2rad(1.915) * std::sin(M_sun) +
                            detail::deg2rad(0.020) * std::sin(2.0 * M_sun);
  const double eps = detail::deg2rad(23.439 - 0.013 * T_jc);
  out_sun_hat[0] = std::cos(lambda_ecl);
  out_sun_hat[1] = std::sin(lambda_ecl) * std::cos(eps);
  out_sun_hat[2] = std::sin(lambda_ecl) * std::sin(eps);
}

double eclipse_factor(const double R_eci[3], const double sun_hat[3]) {
  const double proj = -detail::dot3(R_eci, sun_hat);
  if (proj < 0.0) {
    return 1.0;
  }

  const double dot = detail::dot3(R_eci, sun_hat);
  double axis_projection[3];
  detail::scale3(sun_hat, dot, axis_projection);
  double offset[3];
  detail::sub3(R_eci, axis_projection, offset);
  const double d_perp = detail::norm3(offset);
  if (d_perp < kREarth) {
    return 0.0;
  }
  return 1.0;
}

void dipole_field_eci(const double R_eci[3], double /*t*/, double out_B_eci[3]) {
  const double r = detail::norm3(R_eci);
  if (r == 0.0) {
    detail::zero3(out_B_eci);
    return;
  }

  double r_hat[3];
  detail::normalize3(R_eci, r_hat);
  const double m_hat[3] = {0.0, 0.0, -1.0};
  const double factor = kB0Earth * std::pow(kREarth / r, 3);
  const double dot = detail::dot3(m_hat, r_hat);
  for (int i = 0; i < 3; ++i) {
    out_B_eci[i] = factor * (3.0 * dot * r_hat[i] - m_hat[i]);
  }
}

double atm_density(const double R_eci[3]) {
  const double alt_km = detail::norm3(R_eci) - kREarth;
  return std::max(kAtmRho0 * std::exp(-(alt_km - kAtmH0Km) / kAtmHScale), 0.0);
}

void atmosphere_relative_velocity_eci(
    const double V_sc_eci[3],
    const double R_eci[3],
    double out_v_rel_eci[3]) {
  const double omega_earth[3] = {0.0, 0.0, kOmegaEarth};
  double v_atm[3];
  detail::cross3(omega_earth, R_eci, v_atm);
  detail::sub3(V_sc_eci, v_atm, out_v_rel_eci);
}

void update_environment_cache(
    const OrbitState& orbit,
    const FrameCache& /*frame_cache*/,
    EnvironmentCache* out_cache) {
  if (!out_cache) {
    return;
  }

  sun_vector_eci(orbit.t, out_cache->sun_vector_eci);
  out_cache->eclipse = eclipse_factor(orbit.R_eci, out_cache->sun_vector_eci);
  dipole_field_eci(orbit.R_eci, orbit.t, out_cache->mag_field_eci);
  out_cache->atmosphere_omega_eci[0] = 0.0;
  out_cache->atmosphere_omega_eci[1] = 0.0;
  out_cache->atmosphere_omega_eci[2] = kOmegaEarth;
  out_cache->atm_density = atm_density(orbit.R_eci);
}

}  // namespace mujoco_orbit
