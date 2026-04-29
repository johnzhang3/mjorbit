#ifndef MUJOCO_ORBIT_RUNTIME_H_
#define MUJOCO_ORBIT_RUNTIME_H_

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <mujoco/mujoco.h>

#include "orbit_instance.h"
#include "mujoco_orbit/spec.h"

namespace mujoco_orbit {

struct SensorDescriptor {
  int sensor_id = -1;
  std::string name;
  int sensor_type = 0;
  int datatype = 0;
  int objtype = 0;
  int objid = -1;
  int adr = 0;
  int dim = 0;
  double noise = 0.0;
  double cutoff = 0.0;
  int needstage = 0;
  std::vector<double> user;
  std::string orbit_kind;
  std::array<double, 3> reference_eci = {0.0, 0.0, 0.0};
  std::vector<double> additive_bias_sigma;
  std::vector<double> angular_bias_sigma;
};

struct SensorCatalog {
  std::vector<SensorDescriptor> descriptors;
  std::unordered_map<std::string, SensorDescriptor> by_name;
  std::vector<SensorDescriptor> custom_descriptors;
};

class MjoModel {
 public:
  static std::unique_ptr<MjoModel> FromXmlPath(
      const std::string& xml_path,
      std::optional<double> mj_timestep = std::nullopt);

  MjoModel(const MjoModel&) = delete;
  MjoModel& operator=(const MjoModel&) = delete;
  ~MjoModel();

  mjModel* raw() { return model_; }
  const mjModel* raw() const { return model_; }

  int body_id(const std::string& name) const;
  const SensorDescriptor& sensor(const std::string& name) const;

  int orbit_plugin_instance() const { return orbit_plugin_instance_; }
  const SensorCatalog& sensors() const { return sensors_; }

  const std::vector<SurfaceMetadataNative>& surfaces() const { return surfaces_; }
  const std::vector<MagneticMetadataNative>& magnetic_bodies() const {
    return magnetic_bodies_;
  }
  const std::vector<ReactionWheelMetadataNative>& reaction_wheels() const {
    return reaction_wheels_;
  }
  const std::vector<MagnetorquerMetadataNative>& magnetorquers() const {
    return magnetorquers_;
  }
  const std::vector<ThrusterMetadataNative>& thrusters() const { return thrusters_; }
  const std::vector<ControlMomentGyroMetadataNative>& cmgs() const { return cmgs_; }
  const std::vector<OrbitSensorDescriptorNative>& orbit_sensors() const {
    return orbit_sensors_;
  }
  std::vector<SurfaceMetadataNative>& mutable_surfaces() { return surfaces_; }
  std::vector<MagneticMetadataNative>& mutable_magnetic_bodies() {
    return magnetic_bodies_;
  }
  std::vector<ReactionWheelMetadataNative>& mutable_reaction_wheels() {
    return reaction_wheels_;
  }
  std::vector<MagnetorquerMetadataNative>& mutable_magnetorquers() {
    return magnetorquers_;
  }
  std::vector<ThrusterMetadataNative>& mutable_thrusters() { return thrusters_; }
  std::vector<ControlMomentGyroMetadataNative>& mutable_cmgs() { return cmgs_; }

  bool use_j2() const { return use_j2_; }
  bool use_drag() const { return use_drag_; }
  bool use_srp() const { return use_srp_; }
  bool use_magnetic() const { return use_magnetic_; }
  bool use_gravity_gradient() const { return use_gravity_gradient_; }
  double orbit_dt() const { return orbit_dt_; }
  const CentralBodySpecNative& central_body() const { return central_body_; }

  int nbody() const { return model_->nbody; }
  int nq() const { return model_->nq; }
  int nv() const { return model_->nv; }
  int nu() const { return model_->nu; }
  int nsensordata() const { return model_->nsensordata; }
  int nsensor() const { return model_->nsensor; }
  int ngeom() const { return model_->ngeom; }
  int nplugin() const { return model_->nplugin; }

 private:
  MjoModel() = default;
  static std::unique_ptr<MjoModel> FromSpecXml(
      const std::string& xml,
      const OrbitSpecNative& orbit,
      const AssetMap& assets,
      std::optional<std::string> source_dir = std::nullopt,
      std::optional<double> mj_timestep = std::nullopt);

  mjModel* model_ = nullptr;
  int orbit_plugin_instance_ = -1;
  CentralBodySpecNative central_body_;
  bool use_j2_ = true;
  bool use_drag_ = true;
  bool use_srp_ = true;
  bool use_magnetic_ = true;
  bool use_gravity_gradient_ = true;
  double orbit_dt_ = 0.0;

  std::vector<SurfaceMetadataNative> surfaces_;
  std::vector<MagneticMetadataNative> magnetic_bodies_;
  std::vector<ReactionWheelMetadataNative> reaction_wheels_;
  std::vector<MagnetorquerMetadataNative> magnetorquers_;
  std::vector<ThrusterMetadataNative> thrusters_;
  std::vector<ControlMomentGyroMetadataNative> cmgs_;
  std::vector<OrbitSensorDescriptorNative> orbit_sensors_;
  SensorCatalog sensors_;

  friend class MjoData;
  friend class MjoSpec;
};

class MjoData {
 public:
  MjoData(
      MjoModel& model,
      const std::array<double, 3>& R_eci,
      const std::array<double, 3>& V_eci,
      double t,
      std::optional<std::uint64_t> rng_seed = std::nullopt);
  MjoData(const MjoData&) = delete;
  MjoData& operator=(const MjoData&) = delete;
  ~MjoData();

  MjoModel& model() { return *model_; }
  const MjoModel& model() const { return *model_; }
  mjData* raw() { return data_; }
  const mjData* raw() const { return data_; }
  OrbitInstance* orbit_instance() { return orbit_instance_; }
  const OrbitInstance* orbit_instance() const { return orbit_instance_; }

  void reset(
      const std::optional<std::array<double, 3>>& R_eci = std::nullopt,
      const std::optional<std::array<double, 3>>& V_eci = std::nullopt,
      std::optional<double> t = std::nullopt);
  void clear_wrench_buffer();

  std::array<double, 3> world_position_from_lvlh(const std::array<double, 3>& position_lvlh_m)
      const;
  std::array<double, 3> world_velocity_from_lvlh(
      const std::array<double, 3>& position_lvlh_m,
      const std::array<double, 3>& velocity_lvlh_m_s) const;
  std::array<double, 3> lvlh_position_from_world(const std::array<double, 3>& position_world_m)
      const;
  std::array<double, 3> lvlh_velocity_from_world(
      const std::array<double, 3>& position_world_m,
      const std::array<double, 3>& velocity_world_m_s) const;

  double* rw_speed() { return rw_speed_.data(); }
  double* rw_momentum() { return rw_momentum_.data(); }
  double* rw_torque_cmd() { return rw_torque_cmd_.data(); }
  double* mtq_dipole_cmd() { return mtq_dipole_cmd_.data(); }
  double* thr_force_cmd() { return thr_force_cmd_.data(); }
  double* cmg_gimbal_angle() { return cmg_gimbal_angle_.data(); }
  double* cmg_gimbal_rate_cmd() { return cmg_gimbal_rate_cmd_.data(); }
  double* cmg_rotor_momentum() { return cmg_rotor_momentum_.data(); }
  double* wrench_buffer() { return wrench_buffer_.data(); }

  const std::unordered_map<std::string, std::vector<double>>& sensor_biases() const {
    return sensor_biases_;
  }
  std::vector<double> measure_sensor(const std::string& name, bool noisy);

 private:
  void bind_native_metadata();
  void set_orbit(const std::array<double, 3>& R_eci, const std::array<double, 3>& V_eci, double t);
  void initialize_sensor_biases(std::optional<std::uint64_t> rng_seed);

  MjoModel* model_ = nullptr;
  mjData* data_ = nullptr;
  OrbitInstance* orbit_instance_ = nullptr;

  std::vector<double> rw_speed_;
  std::vector<double> rw_momentum_;
  std::vector<double> rw_torque_cmd_;
  std::vector<double> mtq_dipole_cmd_;
  std::vector<double> thr_force_cmd_;
  std::vector<double> cmg_gimbal_angle_;
  std::vector<double> cmg_gimbal_rate_cmd_;
  std::vector<double> cmg_rotor_momentum_;
  std::vector<double> wrench_buffer_;

  std::unordered_map<std::string, std::vector<double>> sensor_biases_;
  std::uint64_t rng_state_ = 0;
};

void mjo_forward(MjoModel& model, MjoData& data);
void mjo_step(MjoModel& model, MjoData& data);
int mjo_state_size(const MjoModel& model);
int mjo_control_size(const MjoModel& model, unsigned int control_spec);
void mjo_get_state(const MjoModel& model, const MjoData& data, double* out);
void mjo_set_state(const MjoModel& model, MjoData& data, const double* state, int size);
int mjo_rollout_native(
    MjoModel& model,
    MjoData& data,
    int nbatch,
    int nstep,
    unsigned int control_spec,
    int state_size,
    int control_size,
    const double* initial_state,
    const double* initial_warmstart,
    const double* control,
    double* state,
    double* sensordata);

}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_RUNTIME_H_
