#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <utility>

#include "mjorbit/constants.h"
#include "mjorbit/elements.h"
#include "mjorbit/environment.h"
#include "mjorbit/gravity.h"
#include "mjorbit/lvlh.h"
#include "mjorbit/math_utils.h"
#include "mjorbit/orbit_state.h"
#include "mjorbit/propagator.h"

namespace {

using mjorbit::EnvironmentCache;
using mjorbit::FrameCache;
using mjorbit::OrbitState;
using mjorbit::detail::kPi;

int g_failures = 0;

void fail(const std::string& message) {
  std::cerr << "FAIL: " << message << '\n';
  ++g_failures;
}

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    fail(message);
  }
}

void expect_near(double actual, double expected, double tol, const std::string& message) {
  if (std::abs(actual - expected) > tol) {
    fail(message + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected) +
         " tol=" + std::to_string(tol));
  }
}

void expect_rel_near(double actual, double expected, double rel_tol, const std::string& message) {
  const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
  expect_near(actual, expected, rel_tol * scale, message);
}

double norm3(const double v[3]) { return mjorbit::detail::norm3(v); }

double rel_error3(const double left[3], const double right[3]) {
  double delta[3];
  mjorbit::detail::sub3(left, right, delta);
  return norm3(delta) / std::max({1.0, norm3(left), norm3(right)});
}

void test_point_mass_accel() {
  const double r[3] = {mjorbit::kREarth + 400.0, 200.0, -100.0};
  double accel[3];
  mjorbit::point_mass_accel(r, accel);

  const double r_norm = norm3(r);
  expect_rel_near(
      norm3(accel),
      mjorbit::kGmEarth / (r_norm * r_norm),
      1e-12,
      "point_mass_accel magnitude");

  double cross[3];
  mjorbit::detail::cross3(r, accel, cross);
  expect_near(norm3(cross), 0.0, 1e-15, "point_mass_accel radial direction");
  expect_true(mjorbit::detail::dot3(r, accel) < 0.0, "point_mass_accel points inward");
}

void test_j2_accel() {
  const double r = mjorbit::kREarth + 400.0;
  const double factor =
      1.5 * mjorbit::kJ2Earth * mjorbit::kGmEarth * mjorbit::kREarth *
      mjorbit::kREarth / std::pow(r, 5);

  const double equator[3] = {r, 0.0, 0.0};
  double a_equator[3];
  mjorbit::j2_accel(equator, a_equator);
  expect_rel_near(a_equator[0], -factor * r, 1e-12, "j2_accel equator x");
  expect_near(a_equator[1], 0.0, 1e-20, "j2_accel equator y");
  expect_near(a_equator[2], 0.0, 1e-20, "j2_accel equator z");

  const double pole[3] = {0.0, 0.0, r};
  double a_pole[3];
  mjorbit::j2_accel(pole, a_pole);
  expect_near(a_pole[0], 0.0, 1e-20, "j2_accel pole x");
  expect_near(a_pole[1], 0.0, 1e-20, "j2_accel pole y");
  expect_rel_near(a_pole[2], 2.0 * factor * r, 1e-12, "j2_accel pole z");
  expect_true(norm3(a_pole) > norm3(a_equator), "j2_accel pole magnitude exceeds equator");
}

void test_total_accel() {
  const double r[3] = {mjorbit::kREarth + 500.0, 20.0, -30.0};
  double pm[3];
  double j2[3];
  double total[3];
  double no_j2[3];
  mjorbit::point_mass_accel(r, pm);
  mjorbit::j2_accel(r, j2);
  mjorbit::total_accel(r, total, true);
  mjorbit::total_accel(r, no_j2, false);

  for (int i = 0; i < 3; ++i) {
    expect_rel_near(total[i], pm[i] + j2[i], 1e-12, "total_accel includes J2");
    expect_rel_near(no_j2[i], pm[i], 1e-12, "total_accel without J2");
  }
}

void test_propagate_rk4() {
  const double a = mjorbit::kREarth + 400.0;
  double R0[3];
  double V0[3];
  mjorbit::keplerian_to_cartesian(a, 0.0, 0.0, 0.0, 0.0, 0.0, R0, V0);

  OrbitState state{{R0[0], R0[1], R0[2]}, {V0[0], V0[1], V0[2]}, 0.0};
  OrbitState next{};
  const double dt = 1.0;
  mjorbit::propagate_rk4(state, dt, &next, false);

  const double n = std::sqrt(mjorbit::kGmEarth / (a * a * a));
  const double theta = n * dt;
  const double expected_R[3] = {a * std::cos(theta), a * std::sin(theta), 0.0};
  const double expected_V[3] = {-a * n * std::sin(theta), a * n * std::cos(theta), 0.0};

  expect_true(rel_error3(next.R_eci, expected_R) < 1e-10, "propagate_rk4 circular position");
  expect_true(rel_error3(next.V_eci, expected_V) < 1e-10, "propagate_rk4 circular velocity");
  expect_near(next.t, dt, 1e-15, "propagate_rk4 time advance");
}

void test_lvlh() {
  const double a = mjorbit::kREarth + 400.0;
  double R[3];
  double V[3];
  mjorbit::keplerian_to_cartesian(a, 0.0, mjorbit::detail::deg2rad(51.6), 0.0, 0.0, 0.0, R, V);

  FrameCache frame{};
  mjorbit::update_frame_cache(R, V, &frame, false);

  double transpose[9];
  double gram[9];
  mjorbit::detail::mat3_transpose(frame.C_LI, transpose);
  mjorbit::detail::mat3_mul(frame.C_LI, transpose, gram);
  const double identity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  for (int i = 0; i < 9; ++i) {
    expect_near(gram[i], identity[i], 1e-12, "lvlh orthonormality");
  }
  expect_near(mjorbit::detail::mat3_det(frame.C_LI), 1.0, 1e-12, "lvlh determinant");

  double r_hat[3];
  mjorbit::detail::normalize3(R, r_hat);
  expect_near(frame.C_LI[0], r_hat[0], 1e-12, "lvlh radial x");
  expect_near(frame.C_LI[1], r_hat[1], 1e-12, "lvlh radial y");
  expect_near(frame.C_LI[2], r_hat[2], 1e-12, "lvlh radial z");

  const double n = std::sqrt(mjorbit::kGmEarth / (a * a * a));
  expect_near(frame.omega_lvlh[0], 0.0, 1e-12, "lvlh omega x");
  expect_near(frame.omega_lvlh[1], 0.0, 1e-12, "lvlh omega y");
  expect_rel_near(frame.omega_lvlh[2], n, 1e-10, "lvlh omega z");
  expect_near(norm3(frame.omega_dot_lvlh), 0.0, 1e-14, "lvlh omega_dot circular");
}

void test_eclipse() {
  const double sun_hat[3] = {1.0, 0.0, 0.0};
  const double behind[3] = {-(mjorbit::kREarth + 400.0), 0.0, 0.0};
  const double sunward[3] = {mjorbit::kREarth + 400.0, 0.0, 0.0};
  expect_near(mjorbit::eclipse_factor(behind, sun_hat), 0.0, 0.0, "eclipse shadow");
  expect_near(mjorbit::eclipse_factor(sunward, sun_hat), 1.0, 0.0, "eclipse sunlight");
}

void test_dipole_B() {
  const double r = mjorbit::kREarth + 400.0;
  const double equator[3] = {r, 0.0, 0.0};
  const double pole[3] = {0.0, 0.0, r};
  double B_equator[3];
  double B_pole[3];
  mjorbit::dipole_field_eci(equator, 0.0, B_equator);
  mjorbit::dipole_field_eci(pole, 0.0, B_pole);
  expect_rel_near(norm3(B_pole) / norm3(B_equator), 2.0, 1e-12, "dipole pole/equator ratio");
}

void test_keplerian_roundtrip() {
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> a_dist(mjorbit::kREarth + 300.0, mjorbit::kREarth + 5000.0);
  std::uniform_real_distribution<double> e_dist(0.01, 0.4);
  std::uniform_real_distribution<double> inc_dist(0.1, kPi - 0.1);
  std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * kPi);

  for (int i = 0; i < 10; ++i) {
    const double a = a_dist(rng);
    const double e = e_dist(rng);
    const double inc = inc_dist(rng);
    const double raan = angle_dist(rng);
    const double argp = angle_dist(rng);
    const double nu = angle_dist(rng);

    double R0[3];
    double V0[3];
    mjorbit::keplerian_to_cartesian(a, e, inc, raan, argp, nu, R0, V0);

    double a1 = 0.0;
    double e1 = 0.0;
    double inc1 = 0.0;
    double raan1 = 0.0;
    double argp1 = 0.0;
    double nu1 = 0.0;
    mjorbit::cartesian_to_keplerian(R0, V0, &a1, &e1, &inc1, &raan1, &argp1, &nu1);

    double R1[3];
    double V1[3];
    mjorbit::keplerian_to_cartesian(a1, e1, inc1, raan1, argp1, nu1, R1, V1);

    expect_true(rel_error3(R0, R1) < 1e-10, "keplerian roundtrip position");
    expect_true(rel_error3(V0, V1) < 1e-10, "keplerian roundtrip velocity");
  }
}

}  // namespace

int main() {
  const std::pair<const char*, std::function<void()>> tests[] = {
      {"point_mass_accel", test_point_mass_accel},
      {"j2_accel", test_j2_accel},
      {"total_accel", test_total_accel},
      {"propagate_rk4", test_propagate_rk4},
      {"lvlh", test_lvlh},
      {"eclipse", test_eclipse},
      {"dipole_B", test_dipole_B},
      {"keplerian_roundtrip", test_keplerian_roundtrip},
  };

  for (const auto& [name, fn] : tests) {
    const int before = g_failures;
    fn();
    if (g_failures == before) {
      std::cout << "PASS: " << name << '\n';
    }
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " orbit core test(s) failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "All orbit core tests passed\n";
  return EXIT_SUCCESS;
}
