#include "mjorbit/runtime.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <regex>
#include <stdexcept>
#include <string>

#include "mjorbit/math_utils.h"

namespace mjorbit {
namespace {

constexpr char kPluginName[] = "mjorbit.orbit";
constexpr int kOrbitSensorKindSun = 1;
constexpr int kOrbitSensorKindHorizon = 2;
constexpr int kOrbitSensorKindStar = 3;
constexpr int kOrbitSensorKindMagnetometer = 4;

class VfsHolder {
 public:
  explicit VfsHolder(const AssetMap& assets) {
    if (assets.empty()) {
      return;
    }
    mj_defaultVFS(&vfs_);
    active_ = true;
    for (const auto& item : assets) {
      const int result = mj_addBufferVFS(
          &vfs_,
          item.first.c_str(),
          item.second.empty() ? nullptr : item.second.data(),
          static_cast<int>(item.second.size()));
      if (result != 0) {
        throw std::runtime_error("Could not add asset '" + item.first + "' to MuJoCo VFS");
      }
    }
  }

  VfsHolder(const VfsHolder&) = delete;
  VfsHolder& operator=(const VfsHolder&) = delete;

  ~VfsHolder() {
    if (active_) {
      mj_deleteVFS(&vfs_);
    }
  }

  mjVFS* get() { return active_ ? &vfs_ : nullptr; }

 private:
  mjVFS vfs_{};
  bool active_ = false;
};

std::string regex_escape(const std::string& value) {
  static const std::regex special(R"([.^$|()\\[\]{}*+?])");
  return std::regex_replace(value, special, R"(\$&)");
}

class WorkingDirectoryGuard {
 public:
  explicit WorkingDirectoryGuard(const std::optional<std::string>& source_dir) {
    if (!source_dir.has_value() || source_dir->empty()) {
      return;
    }
    previous_ = std::filesystem::current_path();
    std::filesystem::current_path(*source_dir);
    active_ = true;
  }

  WorkingDirectoryGuard(const WorkingDirectoryGuard&) = delete;
  WorkingDirectoryGuard& operator=(const WorkingDirectoryGuard&) = delete;

  ~WorkingDirectoryGuard() {
    if (active_) {
      std::filesystem::current_path(previous_);
    }
  }

 private:
  std::filesystem::path previous_;
  bool active_ = false;
};

void normalized3(const double in[3], double out[3], const std::string& label) {
  const double norm = detail::norm3(in);
  if (norm < 1.0e-12) {
    throw std::runtime_error(label + " must be non-zero");
  }
  detail::scale3(in, 1.0 / norm, out);
}

bool plugin_attrs_name_orbit(const std::string& attrs) {
  const std::regex attr_re(R"ATTR(\bplugin\s*=\s*(?:"([^"]*)"|'([^']*)'))ATTR");
  std::smatch match;
  if (!std::regex_search(attrs, match, attr_re)) {
    return false;
  }
  const std::string value = match[1].matched ? match[1].str() : match[2].str();
  return value == kPluginName;
}

bool text_has_orbit_plugin_tag(const std::string& text) {
  std::size_t pos = 0;
  while ((pos = text.find("<plugin", pos)) != std::string::npos) {
    const std::size_t tag_end = text.find('>', pos);
    if (tag_end == std::string::npos) {
      return false;
    }
    const std::size_t attrs_start = pos + std::string("<plugin").size();
    if (plugin_attrs_name_orbit(text.substr(attrs_start, tag_end - attrs_start))) {
      return true;
    }
    pos = tag_end + 1;
  }
  return false;
}

bool extension_has_orbit_plugin(const std::string& xml) {
  const std::regex extension_re(R"(<extension\b[^>]*>([\s\S]*?)</extension>)");
  for (std::sregex_iterator it(xml.begin(), xml.end(), extension_re), end; it != end; ++it) {
    if (text_has_orbit_plugin_tag((*it)[1].str())) {
      return true;
    }
  }
  return false;
}

bool body_has_direct_orbit_plugin(const std::string& xml, std::size_t body_content_start) {
  int nested_body_depth = 0;
  std::size_t pos = body_content_start;
  while (pos < xml.size()) {
    const std::size_t next_body = xml.find("<body", pos);
    const std::size_t next_plugin = xml.find("<plugin", pos);
    const std::size_t next_body_close = xml.find("</body>", pos);
    const std::size_t next = std::min({next_body, next_plugin, next_body_close});
    if (next == std::string::npos) {
      return false;
    }
    if (next == next_body_close) {
      if (nested_body_depth == 0) {
        return false;
      }
      --nested_body_depth;
      pos = next_body_close + std::string("</body>").size();
      continue;
    }
    const std::size_t tag_end = xml.find('>', next);
    if (tag_end == std::string::npos) {
      return false;
    }
    if (next == next_body) {
      ++nested_body_depth;
      pos = tag_end + 1;
      continue;
    }
    if (nested_body_depth == 0 &&
        text_has_orbit_plugin_tag(xml.substr(next, tag_end - next + 1))) {
      return true;
    }
    pos = tag_end + 1;
  }
  return false;
}

void ensure_extension_plugin(std::string* xml) {
  if (extension_has_orbit_plugin(*xml)) return;
  const std::string plugin = "\n    <plugin plugin=\"mjorbit.orbit\"/>\n";
  const std::size_t extension_close = xml->find("</extension>");
  if (extension_close != std::string::npos) {
    xml->insert(extension_close, plugin);
    return;
  }
  const std::size_t worldbody = xml->find("<worldbody");
  if (worldbody == std::string::npos) {
    throw std::runtime_error("MuJoCo model must define <worldbody>");
  }
  xml->insert(worldbody, "<extension>" + plugin + "  </extension>\n");
}

void ensure_plugin_host(
    std::string* xml,
    const std::optional<std::string>& plugin_body,
    bool use_j2) {
  const std::string plugin =
      "\n      <plugin plugin=\"mjorbit.orbit\">"
      "<config key=\"use_j2\" value=\"" +
      std::string(use_j2 ? "true" : "false") + "\"/></plugin>\n";
  std::regex body_re(R"(<body\b[^>]*>)");
  if (plugin_body.has_value() && !plugin_body->empty()) {
    body_re = std::regex(
        "<body\\b(?=[^>]*\\bname\\s*=\\s*\"" + regex_escape(*plugin_body) + "\")[^>]*>|"
        "<body\\b(?=[^>]*\\bname\\s*=\\s*'" + regex_escape(*plugin_body) + "')[^>]*>");
  }
  std::smatch match;
  if (!std::regex_search(*xml, match, body_re)) {
    if (plugin_body.has_value() && !plugin_body->empty()) {
      throw std::runtime_error("Plugin host body '" + *plugin_body + "' was not found");
    }
    throw std::runtime_error("MuJoCo model must contain at least one body for the orbit plugin");
  }
  const std::size_t insert_at =
      static_cast<std::size_t>(match.position(0) + match.length(0));
  if (body_has_direct_orbit_plugin(*xml, insert_at)) {
    return;
  }
  xml->insert(insert_at, plugin);
}

int resolve_body_id(const mjModel* model, const std::string& name) {
  const int body_id = mj_name2id(model, mjOBJ_BODY, name.c_str());
  if (body_id < 0) {
    throw std::runtime_error("Body '" + name + "' not found in MuJoCo model");
  }
  return body_id;
}

std::vector<double> parse_bias_sigma(const std::vector<double>& user, int dim, int offset = 0) {
  if (offset >= static_cast<int>(user.size())) {
    return {};
  }
  const int available = static_cast<int>(user.size()) - offset;
  std::vector<double> sigma(static_cast<std::size_t>(dim), 0.0);
  bool vector_payload = available >= dim;
  if (vector_payload) {
    bool any_tail = false;
    for (int i = 1; i < dim; ++i) {
      any_tail = any_tail || std::abs(user[offset + i]) > 0.0;
    }
    vector_payload = any_tail;
  }
  if (vector_payload) {
    for (int i = 0; i < dim; ++i) {
      sigma[static_cast<std::size_t>(i)] = std::max(0.0, user[offset + i]);
    }
  } else {
    const double sigma0 = std::max(0.0, user[offset]);
    if (sigma0 <= 0.0) {
      return {};
    }
    std::fill(sigma.begin(), sigma.end(), sigma0);
  }
  if (std::all_of(sigma.begin(), sigma.end(), [](double v) { return v <= 0.0; })) {
    return {};
  }
  return sigma;
}

std::string orbit_sensor_kind(const std::string& name) {
  if (name.rfind("orbit_sun_", 0) == 0) return "sun";
  if (name.rfind("orbit_horizon_", 0) == 0) return "horizon";
  if (name.rfind("orbit_star_", 0) == 0) return "star";
  if (name.rfind("orbit_", 0) == 0) {
    throw std::runtime_error(
        "Unsupported custom sensor '" + name +
        "'. Use orbit_sun_, orbit_horizon_, or orbit_star_.");
  }
  return "";
}

void validate_orbit_sensor(const SensorDescriptor& descriptor) {
  if (descriptor.sensor_type != mjSENS_USER) {
    throw std::runtime_error("Sensor '" + descriptor.name + "' must be a user sensor");
  }
  if (descriptor.objtype != mjOBJ_SITE) {
    throw std::runtime_error("Sensor '" + descriptor.name + "' must attach to a site");
  }
  if (descriptor.datatype != mjDATATYPE_AXIS) {
    throw std::runtime_error("Sensor '" + descriptor.name + "' must use datatype='axis'");
  }
  if (descriptor.needstage != mjSTAGE_POS) {
    throw std::runtime_error("Sensor '" + descriptor.name + "' must use needstage='pos'");
  }
  if (descriptor.dim != 3) {
    throw std::runtime_error("Sensor '" + descriptor.name + "' must declare dim='3'");
  }
}

SensorCatalog compile_sensor_catalog(const mjModel* model) {
  SensorCatalog catalog;
  for (int sensor_id = 0; sensor_id < model->nsensor; ++sensor_id) {
    const char* raw_name = mj_id2name(model, mjOBJ_SENSOR, sensor_id);
    if (!raw_name) {
      throw std::runtime_error("Sensor id " + std::to_string(sensor_id) + " is missing a name");
    }
    SensorDescriptor descriptor;
    descriptor.sensor_id = sensor_id;
    descriptor.name = raw_name;
    descriptor.sensor_type = model->sensor_type[sensor_id];
    descriptor.datatype = model->sensor_datatype[sensor_id];
    descriptor.objtype = model->sensor_objtype[sensor_id];
    descriptor.objid = model->sensor_objid[sensor_id];
    descriptor.adr = model->sensor_adr[sensor_id];
    descriptor.dim = model->sensor_dim[sensor_id];
    descriptor.noise = model->sensor_noise[sensor_id];
    descriptor.cutoff = model->sensor_cutoff[sensor_id];
    descriptor.needstage = model->sensor_needstage[sensor_id];
    descriptor.user.assign(
        model->sensor_user + sensor_id * model->nuser_sensor,
        model->sensor_user + (sensor_id + 1) * model->nuser_sensor);
    descriptor.orbit_kind = orbit_sensor_kind(descriptor.name);

    if (descriptor.sensor_type == mjSENS_ACCELEROMETER ||
        descriptor.sensor_type == mjSENS_GYRO ||
        descriptor.sensor_type == mjSENS_MAGNETOMETER) {
      descriptor.additive_bias_sigma = parse_bias_sigma(descriptor.user, descriptor.dim);
    }

    if (descriptor.orbit_kind == "star") {
      if (descriptor.user.size() < 3) {
        throw std::runtime_error("Sensor '" + descriptor.name + "' requires user[0:3]");
      }
      double ref[3] = {descriptor.user[0], descriptor.user[1], descriptor.user[2]};
      normalized3(ref, descriptor.reference_eci.data(), "Sensor '" + descriptor.name + "' star vector");
      descriptor.angular_bias_sigma = parse_bias_sigma(descriptor.user, 3, 3);
    } else if (descriptor.datatype == mjDATATYPE_AXIS ||
               descriptor.datatype == mjDATATYPE_QUATERNION) {
      descriptor.angular_bias_sigma = parse_bias_sigma(descriptor.user, 3);
    }

    if (!descriptor.orbit_kind.empty()) {
      validate_orbit_sensor(descriptor);
      catalog.custom_descriptors.push_back(descriptor);
    }
    catalog.by_name[descriptor.name] = descriptor;
    catalog.descriptors.push_back(descriptor);
  }
  return catalog;
}

std::vector<OrbitSensorDescriptorNative> build_orbit_sensor_descriptors(
    const SensorCatalog& catalog) {
  std::vector<OrbitSensorDescriptorNative> out;
  for (const SensorDescriptor& descriptor : catalog.custom_descriptors) {
    OrbitSensorDescriptorNative native{};
    native.sensor_id = descriptor.sensor_id;
    native.site_id = descriptor.objid;
    native.adr = descriptor.adr;
    native.dim = descriptor.dim;
    if (descriptor.orbit_kind == "sun") native.kind = kOrbitSensorKindSun;
    if (descriptor.orbit_kind == "horizon") native.kind = kOrbitSensorKindHorizon;
    if (descriptor.orbit_kind == "star") native.kind = kOrbitSensorKindStar;
    for (int i = 0; i < 3; ++i) native.reference_eci[i] = descriptor.reference_eci[i];
    out.push_back(native);
  }
  for (const SensorDescriptor& descriptor : catalog.descriptors) {
    if (descriptor.sensor_type != mjSENS_MAGNETOMETER) {
      continue;
    }
    OrbitSensorDescriptorNative native{};
    native.sensor_id = descriptor.sensor_id;
    native.kind = kOrbitSensorKindMagnetometer;
    native.site_id = descriptor.objid;
    native.adr = descriptor.adr;
    native.dim = descriptor.dim;
    out.push_back(native);
  }
  return out;
}

int resolve_plugin_body_id(const mjModel* model, const OrbitSpecNative& orbit) {
  if (orbit.plugin_body.has_value() && !orbit.plugin_body->empty()) {
    return resolve_body_id(model, *orbit.plugin_body);
  }
  if (model->nbody <= 1) {
    throw std::runtime_error("MuJoCo model must contain at least one body for the orbit plugin");
  }
  return 1;
}

void resolve_config(const OrbitSpecNative& parsed, MjoModel* model) {
  const mjModel* mjm = model->raw();
  for (const OrbitSurfaceSpecNative& spec : parsed.surfaces) {
    if (spec.area <= 0.0) {
      throw std::runtime_error("Surface '" + spec.name + "' area must be positive");
    }
    if (spec.drag_coeff < 0.0) {
      throw std::runtime_error("Surface '" + spec.name + "' drag_coeff must be non-negative");
    }
    if (spec.srp_coeff < 0.0) {
      throw std::runtime_error("Surface '" + spec.name + "' srp_coeff must be non-negative");
    }
    SurfaceMetadataNative native{};
    native.body_id = resolve_body_id(mjm, spec.body_name);
    detail::copy3(spec.center_of_pressure_body.data(), native.center_of_pressure_body);
    normalized3(spec.normal_body.data(), native.normal_body, "Surface '" + spec.name + "' normal");
    native.area = spec.area;
    native.drag_coeff = spec.drag_coeff;
    native.srp_coeff = spec.srp_coeff;
    native.use_drag = spec.use_drag ? 1 : 0;
    native.use_srp = spec.use_srp ? 1 : 0;
    model->mutable_surfaces().push_back(native);
  }
  for (const OrbitMagneticBodySpecNative& spec : parsed.magnetic_bodies) {
    MagneticMetadataNative native{};
    native.body_id = resolve_body_id(mjm, spec.body_name);
    detail::copy3(spec.dipole_body.data(), native.dipole_body);
    model->mutable_magnetic_bodies().push_back(native);
  }
  for (const OrbitReactionWheelSpecNative& spec : parsed.reaction_wheels) {
    if (spec.inertia <= 0.0) {
      throw std::runtime_error("Reaction wheel '" + spec.name + "' inertia must be positive");
    }
    if (spec.speed_limit.has_value() && *spec.speed_limit <= 0.0) {
      throw std::runtime_error(
          "Reaction wheel '" + spec.name + "' speed_limit must be positive");
    }
    if (spec.torque_limit.has_value() && *spec.torque_limit <= 0.0) {
      throw std::runtime_error(
          "Reaction wheel '" + spec.name + "' torque_limit must be positive");
    }
    ReactionWheelMetadataNative native{};
    native.body_id = resolve_body_id(mjm, spec.body_name);
    normalized3(spec.axis_body.data(), native.axis_body, "Reaction wheel '" + spec.name + "' axis");
    native.inertia = spec.inertia;
    native.has_speed_limit = spec.speed_limit.has_value() ? 1 : 0;
    native.speed_limit = spec.speed_limit.value_or(0.0);
    native.has_torque_limit = spec.torque_limit.has_value() ? 1 : 0;
    native.torque_limit = spec.torque_limit.value_or(0.0);
    model->mutable_reaction_wheels().push_back(native);
  }
  for (const OrbitMagnetorquerSpecNative& spec : parsed.magnetorquers) {
    if (spec.dipole_limit <= 0.0) {
      throw std::runtime_error("Magnetorquer '" + spec.name + "' dipole_limit must be positive");
    }
    MagnetorquerMetadataNative native{};
    native.body_id = resolve_body_id(mjm, spec.body_name);
    normalized3(spec.axis_body.data(), native.axis_body, "Magnetorquer '" + spec.name + "' axis");
    native.dipole_limit = spec.dipole_limit;
    model->mutable_magnetorquers().push_back(native);
  }
  for (const OrbitThrusterSpecNative& spec : parsed.thrusters) {
    if (spec.force_limit <= 0.0) {
      throw std::runtime_error("Thruster '" + spec.name + "' force_limit must be positive");
    }
    ThrusterMetadataNative native{};
    native.body_id = resolve_body_id(mjm, spec.body_name);
    detail::copy3(spec.position_body.data(), native.position_body);
    normalized3(spec.direction_body.data(), native.direction_body, "Thruster '" + spec.name + "' direction");
    native.force_limit = spec.force_limit;
    model->mutable_thrusters().push_back(native);
  }
  for (const OrbitCmgSpecNative& spec : parsed.cmgs) {
    ControlMomentGyroMetadataNative native{};
    native.body_id = resolve_body_id(mjm, spec.body_name);
    normalized3(spec.gimbal_axis_body.data(), native.gimbal_axis_body, "CMG '" + spec.name + "' gimbal axis");
    normalized3(spec.spin_axis_body_0.data(), native.spin_axis_body_0, "CMG '" + spec.name + "' spin axis");
    const double dot = detail::dot3(native.gimbal_axis_body, native.spin_axis_body_0);
    if (std::abs(dot) > 1.0e-8) {
      throw std::runtime_error("CMG '" + spec.name + "' spin axis must be orthogonal to gimbal axis");
    }
    if (spec.rotor_momentum <= 0.0) {
      throw std::runtime_error("CMG '" + spec.name + "' rotor_momentum must be positive");
    }
    if (spec.gimbal_rate_limit.has_value() && *spec.gimbal_rate_limit <= 0.0) {
      throw std::runtime_error("CMG '" + spec.name + "' gimbal_rate_limit must be positive");
    }
    if (spec.gimbal_angle_limit.has_value() && *spec.gimbal_angle_limit <= 0.0) {
      throw std::runtime_error("CMG '" + spec.name + "' gimbal_angle_limit must be positive");
    }
    detail::cross3(native.gimbal_axis_body, native.spin_axis_body_0, native.torque_axis_body_0);
    native.rotor_momentum = spec.rotor_momentum;
    native.has_gimbal_rate_limit = spec.gimbal_rate_limit.has_value() ? 1 : 0;
    native.gimbal_rate_limit = spec.gimbal_rate_limit.value_or(0.0);
    native.has_gimbal_angle_limit = spec.gimbal_angle_limit.has_value() ? 1 : 0;
    native.gimbal_angle_limit = spec.gimbal_angle_limit.value_or(0.0);
    model->mutable_cmgs().push_back(native);
  }
}

}  // namespace

MjoModel::~MjoModel() {
  if (model_) {
    mj_deleteModel(model_);
    model_ = nullptr;
  }
}

std::unique_ptr<MjoModel> MjoModel::FromXmlPath(
    const std::string& xml_path,
    std::optional<double> mj_timestep) {
  return MjoSpec::FromXmlPath(xml_path)->Compile(mj_timestep);
}

std::unique_ptr<MjoModel> MjoModel::FromSpecXml(
    const std::string& xml,
    const OrbitSpecNative& orbit,
    const AssetMap& assets,
    std::optional<std::string> source_dir,
    std::optional<double> mj_timestep) {
  std::string compiled_xml = xml;
  ensure_extension_plugin(&compiled_xml);
  ensure_plugin_host(&compiled_xml, orbit.plugin_body, orbit.use_j2);

  char error[2048] = {0};
  VfsHolder vfs(assets);
  WorkingDirectoryGuard cwd(source_dir);
  mjSpec* compiled_spec =
      mj_parseXMLString(compiled_xml.c_str(), vfs.get(), error, sizeof(error));
  if (!compiled_spec) {
    throw std::runtime_error(std::string("MuJoCo XML parse failed: ") + error);
  }
  mjModel* raw_model = mj_compile(compiled_spec, vfs.get());
  if (!raw_model) {
    const char* compile_error = mjs_getError(compiled_spec);
    std::string message = compile_error && compile_error[0] != '\0'
                              ? compile_error
                              : "unknown compiler error";
    mj_deleteSpec(compiled_spec);
    throw std::runtime_error("MuJoCo XML compile failed: " + message);
  }
  mj_deleteSpec(compiled_spec);

  std::unique_ptr<MjoModel> model(new MjoModel());
  model->model_ = raw_model;
  if (mj_timestep.has_value()) {
    model->model_->opt.timestep = *mj_timestep;
  }
  model->model_->opt.gravity[0] = 0.0;
  model->model_->opt.gravity[1] = 0.0;
  model->model_->opt.gravity[2] = 0.0;
  model->central_body_ = orbit.central_body;
  model->use_j2_ = orbit.use_j2;
  model->use_drag_ = orbit.use_drag;
  model->use_srp_ = orbit.use_srp;
  model->use_magnetic_ = orbit.use_magnetic;
  model->use_gravity_gradient_ = orbit.use_gravity_gradient;
  model->orbit_dt_ = orbit.orbit_dt.value_or(0.0);

  const int plugin_body_id = resolve_plugin_body_id(model->model_, orbit);
  model->orbit_plugin_instance_ = model->model_->body_plugin[plugin_body_id];
  if (model->orbit_plugin_instance_ < 0) {
    throw std::runtime_error("Failed to attach mjorbit plugin to host body");
  }

  resolve_config(orbit, model.get());
  model->sensors_ = compile_sensor_catalog(model->model_);
  model->orbit_sensors_ = build_orbit_sensor_descriptors(model->sensors_);
  return model;
}

int MjoModel::body_id(const std::string& name) const { return resolve_body_id(model_, name); }

const SensorDescriptor& MjoModel::sensor(const std::string& name) const {
  auto it = sensors_.by_name.find(name);
  if (it == sensors_.by_name.end()) {
    throw std::runtime_error("Sensor '" + name + "' not found in model");
  }
  return it->second;
}

}  // namespace mjorbit
