#ifndef MJORBIT_SPEC_H_
#define MJORBIT_SPEC_H_

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjorbit/constants.h"

namespace mjorbit {

using AssetMap = std::unordered_map<std::string, std::vector<std::uint8_t>>;

struct CentralBodySpecNative {
  std::string name = "earth";
  double gm = kGmEarth;
  double radius = kREarth;
  double j2 = kJ2Earth;
  std::array<double, 3> omega = {0.0, 0.0, kOmegaEarth};
  double magnetic_b0 = kB0Earth;
  std::array<double, 3> magnetic_axis = {0.0, 0.0, -1.0};
  double atmosphere_h0 = 400.0;
  double atmosphere_rho0 = 2.62e-13;
  double atmosphere_scale_height = 58.2;
};

struct OrbitSurfaceSpecNative {
  std::string name;
  std::string body_name;
  std::array<double, 3> center_of_pressure_body = {0.0, 0.0, 0.0};
  std::array<double, 3> normal_body = {0.0, 0.0, 1.0};
  double area = 0.0;
  double drag_coeff = 2.2;
  double srp_coeff = 1.8;
  bool use_drag = true;
  bool use_srp = true;
};

struct OrbitMagneticBodySpecNative {
  std::string name;
  std::string body_name;
  std::array<double, 3> dipole_body = {0.0, 0.0, 0.0};
};

struct OrbitReactionWheelSpecNative {
  std::string name;
  std::string body_name;
  std::array<double, 3> axis_body = {0.0, 0.0, 1.0};
  double inertia = 0.0;
  std::optional<double> speed_limit;
  std::optional<double> torque_limit;
};

struct OrbitMagnetorquerSpecNative {
  std::string name;
  std::string body_name;
  std::array<double, 3> axis_body = {0.0, 0.0, 1.0};
  double dipole_limit = 0.0;
};

struct OrbitThrusterSpecNative {
  std::string name;
  std::string body_name;
  std::array<double, 3> position_body = {0.0, 0.0, 0.0};
  std::array<double, 3> direction_body = {1.0, 0.0, 0.0};
  double force_limit = 0.0;
};

struct OrbitCmgSpecNative {
  std::string name;
  std::string body_name;
  std::array<double, 3> gimbal_axis_body = {0.0, 1.0, 0.0};
  std::array<double, 3> spin_axis_body_0 = {0.0, 0.0, 1.0};
  double rotor_momentum = 0.0;
  std::optional<double> gimbal_rate_limit;
  std::optional<double> gimbal_angle_limit;
};

struct OrbitSpecNative {
  std::optional<std::string> plugin_body;
  bool use_j2 = true;
  bool use_drag = true;
  bool use_srp = true;
  bool use_magnetic = true;
  bool use_gravity_gradient = true;
  std::optional<double> orbit_dt;
  CentralBodySpecNative central_body;
  std::vector<OrbitSurfaceSpecNative> surfaces;
  std::vector<OrbitMagneticBodySpecNative> magnetic_bodies;
  std::vector<OrbitReactionWheelSpecNative> reaction_wheels;
  std::vector<OrbitMagnetorquerSpecNative> magnetorquers;
  std::vector<OrbitThrusterSpecNative> thrusters;
  std::vector<OrbitCmgSpecNative> cmgs;
};

class MjoModel;

class MjoSpec {
 public:
  static std::unique_ptr<MjoSpec> FromXmlPath(const std::string& xml_path);
  static std::unique_ptr<MjoSpec> FromXmlString(
      const std::string& xml,
      AssetMap assets = {});

  MjoSpec() = default;
  MjoSpec(const MjoSpec&) = delete;
  MjoSpec& operator=(const MjoSpec&) = delete;
  ~MjoSpec();

  std::unique_ptr<MjoSpec> Copy() const;
  std::unique_ptr<MjoModel> Compile(std::optional<double> mj_timestep = std::nullopt) const;
  std::string ToXml() const;

  const OrbitSpecNative& orbit() const { return orbit_; }
  OrbitSpecNative& orbit() { return orbit_; }

  std::optional<std::string> plugin_body() const { return orbit_.plugin_body; }
  void set_plugin_body(std::optional<std::string> value) { orbit_.plugin_body = std::move(value); }
  bool use_j2() const { return orbit_.use_j2; }
  void set_use_j2(bool value) { orbit_.use_j2 = value; }
  bool use_drag() const { return orbit_.use_drag; }
  void set_use_drag(bool value) { orbit_.use_drag = value; }
  bool use_srp() const { return orbit_.use_srp; }
  void set_use_srp(bool value) { orbit_.use_srp = value; }
  bool use_magnetic() const { return orbit_.use_magnetic; }
  void set_use_magnetic(bool value) { orbit_.use_magnetic = value; }
  bool use_gravity_gradient() const { return orbit_.use_gravity_gradient; }
  void set_use_gravity_gradient(bool value) { orbit_.use_gravity_gradient = value; }
  std::optional<double> orbit_dt() const { return orbit_.orbit_dt; }
  void set_orbit_dt(std::optional<double> value) { orbit_.orbit_dt = value; }
  const CentralBodySpecNative& central_body() const { return orbit_.central_body; }
  void set_central_body(const CentralBodySpecNative& value) { orbit_.central_body = value; }

  std::string AddSurface(OrbitSurfaceSpecNative spec);
  void SetSurface(const std::string& name, OrbitSurfaceSpecNative spec);
  void RemoveSurface(const std::string& name);
  std::string AddMagneticBody(OrbitMagneticBodySpecNative spec);
  void SetMagneticBody(const std::string& name, OrbitMagneticBodySpecNative spec);
  void RemoveMagneticBody(const std::string& name);
  std::string AddReactionWheel(OrbitReactionWheelSpecNative spec);
  void SetReactionWheel(const std::string& name, OrbitReactionWheelSpecNative spec);
  void RemoveReactionWheel(const std::string& name);
  std::string AddMagnetorquer(OrbitMagnetorquerSpecNative spec);
  void SetMagnetorquer(const std::string& name, OrbitMagnetorquerSpecNative spec);
  void RemoveMagnetorquer(const std::string& name);
  std::string AddThruster(OrbitThrusterSpecNative spec);
  void SetThruster(const std::string& name, OrbitThrusterSpecNative spec);
  void RemoveThruster(const std::string& name);
  std::string AddCmg(OrbitCmgSpecNative spec);
  void SetCmg(const std::string& name, OrbitCmgSpecNative spec);
  void RemoveCmg(const std::string& name);

 private:
  mjSpec* spec_ = nullptr;
  std::string xml_;
  OrbitSpecNative orbit_;
  AssetMap assets_;
  std::optional<std::string> source_dir_;
};

}  // namespace mjorbit

#endif  // MJORBIT_SPEC_H_
