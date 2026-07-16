#ifndef MJORBIT_ENVIRONMENT_H_
#define MJORBIT_ENVIRONMENT_H_

#include "mjorbit/orbit_state.h"
#include "mjorbit/spec.h"

namespace mjorbit {

void sun_vector_eci(double t, double out_sun_hat[3]);

double eclipse_factor(const double R_eci[3], const double sun_hat[3]);
double eclipse_factor(
    const double R_eci[3],
    const double sun_hat[3],
    const CentralBodySpecNative& central_body);

// Direction of the central body's magnetic dipole axis in ECI at time t
// (seconds since J2000.0). The configured magnetic_axis is body-fixed (ECEF
// components) and co-rotates about the spin axis omega with the Earth Rotation
// Angle phase; with the default axis parallel to omega this is the identity.
void magnetic_axis_eci(
    double t, double out_m_hat[3], const CentralBodySpecNative& central_body);

void dipole_field_eci(const double R_eci[3], double t, double out_B_eci[3]);
void dipole_field_eci(
    const double R_eci[3],
    double t,
    double out_B_eci[3],
    const CentralBodySpecNative& central_body);

double atm_density(const double R_eci[3]);
double atm_density(const double R_eci[3], const CentralBodySpecNative& central_body);

void atmosphere_relative_velocity_eci(
    const double V_sc_eci[3],
    const double R_eci[3],
    double out_v_rel_eci[3]);
void atmosphere_relative_velocity_eci(
    const double V_sc_eci[3],
    const double R_eci[3],
    double out_v_rel_eci[3],
    const CentralBodySpecNative& central_body);

void update_environment_cache(
    const OrbitState& orbit,
    const FrameCache& frame_cache,
    EnvironmentCache* out_cache);
void update_environment_cache(
    const OrbitState& orbit,
    const FrameCache& frame_cache,
    EnvironmentCache* out_cache,
    const CentralBodySpecNative& central_body);

}  // namespace mjorbit

#endif  // MJORBIT_ENVIRONMENT_H_
