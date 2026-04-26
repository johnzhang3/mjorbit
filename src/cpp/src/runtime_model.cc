#include "mujoco_orbit/runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "mujoco_orbit/math_utils.h"

namespace mujoco_orbit {
namespace {

constexpr char kPluginName[] = "mujoco_orbit.orbit";
constexpr char kGeneratedPluginHostName[] = "__mujoco_orbit_plugin_host__";
constexpr int kOrbitSensorKindSun = 1;
constexpr int kOrbitSensorKindHorizon = 2;
constexpr int kOrbitSensorKindStar = 3;
constexpr int kOrbitSensorKindMagnetometer = 4;

struct RawElement {
  std::string tag;
  std::unordered_map<std::string, std::string> attrs;
};

struct SurfaceSpecXml {
  std::string body_name;
  double cop[3] = {0.0, 0.0, 0.0};
  double normal[3] = {0.0, 0.0, 0.0};
  double area = 0.0;
  double drag_coeff = 2.2;
  double srp_coeff = 1.8;
  bool use_drag = true;
  bool use_srp = true;
};

struct MagneticSpecXml {
  std::string body_name;
  double dipole[3] = {0.0, 0.0, 0.0};
};

struct ReactionWheelSpecXml {
  std::string body_name;
  double axis[3] = {0.0, 0.0, 0.0};
  double inertia = 0.0;
  std::optional<double> speed_limit;
  std::optional<double> torque_limit;
};

struct MagnetorquerSpecXml {
  std::string body_name;
  double axis[3] = {0.0, 0.0, 0.0};
  double dipole_limit = 0.0;
};

struct ThrusterSpecXml {
  std::string body_name;
  double position[3] = {0.0, 0.0, 0.0};
  double direction[3] = {0.0, 0.0, 0.0};
  double force_limit = 0.0;
};

struct CmgSpecXml {
  std::string body_name;
  double gimbal_axis[3] = {0.0, 0.0, 0.0};
  double spin_axis0[3] = {0.0, 0.0, 0.0};
  double rotor_momentum = 0.0;
  std::optional<double> gimbal_rate_limit;
  std::optional<double> gimbal_angle_limit;
};

struct ParsedOrbitXml {
  std::string plugin_body;
  bool use_j2 = true;
  bool use_drag = true;
  bool use_srp = true;
  bool use_magnetic = true;
  bool use_gravity_gradient = true;
  double orbit_dt = 0.0;
  std::vector<SurfaceSpecXml> surfaces;
  std::vector<MagneticSpecXml> magnetic_bodies;
  std::vector<ReactionWheelSpecXml> reaction_wheels;
  std::vector<MagnetorquerSpecXml> magnetorquers;
  std::vector<ThrusterSpecXml> thrusters;
  std::vector<CmgSpecXml> cmgs;
};

std::string read_file(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Could not open XML file: " + path);
  }
  std::ostringstream out;
  out << input.rdbuf();
  return out.str();
}

std::string regex_escape(const std::string& value) {
  static const std::regex special(R"([.^$|()\\[\]{}*+?])");
  return std::regex_replace(value, special, R"(\$&)");
}

std::unordered_map<std::string, std::string> parse_attrs(const std::string& text) {
  std::unordered_map<std::string, std::string> attrs;
  const std::regex attr_re(R"ATTR(([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*"([^"]*)")ATTR");
  for (std::sregex_iterator it(text.begin(), text.end(), attr_re), end; it != end; ++it) {
    attrs[(*it)[1].str()] = (*it)[2].str();
  }
  return attrs;
}

bool parse_bool(const std::unordered_map<std::string, std::string>& attrs,
                const std::string& key,
                bool default_value) {
  auto it = attrs.find(key);
  if (it == attrs.end()) {
    return default_value;
  }
  const std::string& value = it->second;
  return !(value == "false" || value == "False" || value == "0" || value == "no");
}

double parse_double(const std::unordered_map<std::string, std::string>& attrs,
                    const std::string& key,
                    double default_value) {
  auto it = attrs.find(key);
  if (it == attrs.end()) {
    return default_value;
  }
  return std::stod(it->second);
}

std::optional<double> parse_optional_double(
    const std::unordered_map<std::string, std::string>& attrs,
    const std::string& key) {
  auto it = attrs.find(key);
  if (it == attrs.end() || it->second.empty()) {
    return std::nullopt;
  }
  return std::stod(it->second);
}

std::string parse_required_string(
    const std::unordered_map<std::string, std::string>& attrs,
    const std::string& key,
    const std::string& tag) {
  auto it = attrs.find(key);
  if (it == attrs.end() || it->second.empty()) {
    throw std::runtime_error("<" + tag + "> requires attribute '" + key + "'");
  }
  return it->second;
}

void parse_vec3(const std::string& text, double out[3], const std::string& label) {
  std::istringstream input(text);
  if (!(input >> out[0] >> out[1] >> out[2])) {
    throw std::runtime_error(label + " must contain three numeric values");
  }
}

void parse_required_vec3(
    const std::unordered_map<std::string, std::string>& attrs,
    const std::string& key,
    const std::string& tag,
    double out[3]) {
  auto it = attrs.find(key);
  if (it == attrs.end()) {
    throw std::runtime_error("<" + tag + "> requires attribute '" + key + "'");
  }
  parse_vec3(it->second, out, "<" + tag + "> " + key);
}

void normalized3(const double in[3], double out[3], const std::string& label) {
  const double norm = detail::norm3(in);
  if (norm < 1.0e-12) {
    throw std::runtime_error(label + " must be non-zero");
  }
  detail::scale3(in, 1.0 / norm, out);
}

std::vector<RawElement> parse_mjorbit_children(const std::string& body) {
  std::vector<RawElement> children;
  const std::regex child_re(R"(<([A-Za-z_][A-Za-z0-9_\-]*)\b([^>]*)/>)");
  for (std::sregex_iterator it(body.begin(), body.end(), child_re), end; it != end; ++it) {
    children.push_back(RawElement{(*it)[1].str(), parse_attrs((*it)[2].str())});
  }
  return children;
}

ParsedOrbitXml parse_mjorbit_block(std::string* xml) {
  ParsedOrbitXml parsed;
  const std::regex block_re(R"(<mjorbit\b([^>]*)>([\s\S]*?)</mjorbit>)");
  std::smatch match;
  if (!std::regex_search(*xml, match, block_re)) {
    return parsed;
  }

  const auto attrs = parse_attrs(match[1].str());
  auto plugin_body_it = attrs.find("plugin_body");
  if (plugin_body_it != attrs.end()) {
    parsed.plugin_body = plugin_body_it->second;
  }
  parsed.use_j2 = parse_bool(attrs, "use_j2", true);
  parsed.use_drag = parse_bool(attrs, "use_drag", true);
  parsed.use_srp = parse_bool(attrs, "use_srp", true);
  parsed.use_magnetic = parse_bool(attrs, "use_magnetic", true);
  parsed.use_gravity_gradient = parse_bool(attrs, "use_gravity_gradient", true);
  parsed.orbit_dt = parse_double(attrs, "orbit_dt", 0.0);

  for (const RawElement& child : parse_mjorbit_children(match[2].str())) {
    if (child.tag == "surface") {
      SurfaceSpecXml spec;
      spec.body_name = parse_required_string(child.attrs, "body", child.tag);
      parse_required_vec3(child.attrs, "cop", child.tag, spec.cop);
      parse_required_vec3(child.attrs, "normal", child.tag, spec.normal);
      spec.area = parse_double(child.attrs, "area", spec.area);
      spec.drag_coeff = parse_double(child.attrs, "drag_coeff", spec.drag_coeff);
      spec.srp_coeff = parse_double(child.attrs, "srp_coeff", spec.srp_coeff);
      spec.use_drag = parse_bool(child.attrs, "use_drag", true);
      spec.use_srp = parse_bool(child.attrs, "use_srp", true);
      parsed.surfaces.push_back(spec);
    } else if (child.tag == "magnetic_body") {
      MagneticSpecXml spec;
      spec.body_name = parse_required_string(child.attrs, "body", child.tag);
      parse_required_vec3(child.attrs, "dipole", child.tag, spec.dipole);
      parsed.magnetic_bodies.push_back(spec);
    } else if (child.tag == "reaction_wheel") {
      ReactionWheelSpecXml spec;
      spec.body_name = parse_required_string(child.attrs, "body", child.tag);
      parse_required_vec3(child.attrs, "axis", child.tag, spec.axis);
      spec.inertia = parse_double(child.attrs, "inertia", spec.inertia);
      spec.speed_limit = parse_optional_double(child.attrs, "speed_limit");
      spec.torque_limit = parse_optional_double(child.attrs, "torque_limit");
      parsed.reaction_wheels.push_back(spec);
    } else if (child.tag == "magnetorquer") {
      MagnetorquerSpecXml spec;
      spec.body_name = parse_required_string(child.attrs, "body", child.tag);
      parse_required_vec3(child.attrs, "axis", child.tag, spec.axis);
      spec.dipole_limit = parse_double(child.attrs, "dipole_limit", spec.dipole_limit);
      parsed.magnetorquers.push_back(spec);
    } else if (child.tag == "thruster") {
      ThrusterSpecXml spec;
      spec.body_name = parse_required_string(child.attrs, "body", child.tag);
      parse_required_vec3(child.attrs, "pos", child.tag, spec.position);
      parse_required_vec3(child.attrs, "dir", child.tag, spec.direction);
      spec.force_limit = parse_double(child.attrs, "force_limit", spec.force_limit);
      parsed.thrusters.push_back(spec);
    } else if (child.tag == "cmg") {
      CmgSpecXml spec;
      spec.body_name = parse_required_string(child.attrs, "body", child.tag);
      parse_required_vec3(child.attrs, "gimbal_axis", child.tag, spec.gimbal_axis);
      parse_required_vec3(child.attrs, "spin_axis0", child.tag, spec.spin_axis0);
      spec.rotor_momentum = parse_double(child.attrs, "rotor_momentum", spec.rotor_momentum);
      spec.gimbal_rate_limit = parse_optional_double(child.attrs, "gimbal_rate_limit");
      spec.gimbal_angle_limit = parse_optional_double(child.attrs, "gimbal_angle_limit");
      parsed.cmgs.push_back(spec);
    }
  }

  xml->erase(static_cast<std::size_t>(match.position(0)), static_cast<std::size_t>(match.length(0)));
  return parsed;
}

std::string first_body_name(const std::string& xml) {
  const std::regex body_re(R"(<body\b([^>]*)>)");
  std::smatch match;
  if (!std::regex_search(xml, match, body_re)) {
    throw std::runtime_error("MuJoCo model must contain at least one body for the orbit plugin");
  }
  auto attrs = parse_attrs(match[1].str());
  auto it = attrs.find("name");
  if (it == attrs.end() || it->second.empty()) {
    return kGeneratedPluginHostName;
  }
  return it->second;
}

void ensure_extension_plugin(std::string* xml) {
  if (xml->find("plugin=\"" + std::string(kPluginName) + "\"") != std::string::npos) {
    return;
  }
  const std::string plugin = "\n    <plugin plugin=\"mujoco_orbit.orbit\"/>\n";
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

void ensure_plugin_host(std::string* xml, const std::string& plugin_body, bool use_j2) {
  const std::string plugin =
      "\n      <plugin plugin=\"mujoco_orbit.orbit\">"
      "<config key=\"use_j2\" value=\"" +
      std::string(use_j2 ? "true" : "false") + "\"/></plugin>\n";
  const std::regex body_re(
      "<body\\b(?=[^>]*\\bname\\s*=\\s*\"" + regex_escape(plugin_body) + "\")[^>]*>");
  std::smatch match;
  if (!std::regex_search(*xml, match, body_re)) {
    throw std::runtime_error("Plugin host body '" + plugin_body + "' was not found");
  }
  const std::size_t insert_at =
      static_cast<std::size_t>(match.position(0) + match.length(0));
  xml->insert(insert_at, plugin);
}

std::string write_temp_xml(const std::string& xml) {
  const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("mujoco_orbit_" + std::to_string(now) + ".xml");
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not create temporary MJCF file");
  }
  out << xml;
  return path.string();
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

void resolve_config(const ParsedOrbitXml& parsed, MjoModel* model) {
  const mjModel* mjm = model->raw();
  for (const SurfaceSpecXml& spec : parsed.surfaces) {
    SurfaceMetadataNative native{};
    native.body_id = resolve_body_id(mjm, spec.body_name);
    detail::copy3(spec.cop, native.center_of_pressure_body);
    normalized3(spec.normal, native.normal_body, "Surface '" + spec.body_name + "' normal");
    native.area = spec.area;
    native.drag_coeff = spec.drag_coeff;
    native.srp_coeff = spec.srp_coeff;
    native.use_drag = spec.use_drag ? 1 : 0;
    native.use_srp = spec.use_srp ? 1 : 0;
    model->mutable_surfaces().push_back(native);
  }
  for (const MagneticSpecXml& spec : parsed.magnetic_bodies) {
    MagneticMetadataNative native{};
    native.body_id = resolve_body_id(mjm, spec.body_name);
    detail::copy3(spec.dipole, native.dipole_body);
    model->mutable_magnetic_bodies().push_back(native);
  }
  for (const ReactionWheelSpecXml& spec : parsed.reaction_wheels) {
    ReactionWheelMetadataNative native{};
    native.body_id = resolve_body_id(mjm, spec.body_name);
    normalized3(spec.axis, native.axis_body, "Reaction wheel '" + spec.body_name + "' axis");
    native.inertia = spec.inertia;
    native.has_speed_limit = spec.speed_limit.has_value() ? 1 : 0;
    native.speed_limit = spec.speed_limit.value_or(0.0);
    native.has_torque_limit = spec.torque_limit.has_value() ? 1 : 0;
    native.torque_limit = spec.torque_limit.value_or(0.0);
    model->mutable_reaction_wheels().push_back(native);
  }
  for (const MagnetorquerSpecXml& spec : parsed.magnetorquers) {
    MagnetorquerMetadataNative native{};
    native.body_id = resolve_body_id(mjm, spec.body_name);
    normalized3(spec.axis, native.axis_body, "Magnetorquer '" + spec.body_name + "' axis");
    native.dipole_limit = spec.dipole_limit;
    model->mutable_magnetorquers().push_back(native);
  }
  for (const ThrusterSpecXml& spec : parsed.thrusters) {
    ThrusterMetadataNative native{};
    native.body_id = resolve_body_id(mjm, spec.body_name);
    detail::copy3(spec.position, native.position_body);
    normalized3(spec.direction, native.direction_body, "Thruster '" + spec.body_name + "' direction");
    native.force_limit = spec.force_limit;
    model->mutable_thrusters().push_back(native);
  }
  for (const CmgSpecXml& spec : parsed.cmgs) {
    ControlMomentGyroMetadataNative native{};
    native.body_id = resolve_body_id(mjm, spec.body_name);
    normalized3(spec.gimbal_axis, native.gimbal_axis_body, "CMG '" + spec.body_name + "' gimbal axis");
    normalized3(spec.spin_axis0, native.spin_axis_body_0, "CMG '" + spec.body_name + "' spin axis");
    const double dot = detail::dot3(native.gimbal_axis_body, native.spin_axis_body_0);
    if (std::abs(dot) > 1.0e-8) {
      throw std::runtime_error("CMG '" + spec.body_name + "' spin axis must be orthogonal to gimbal axis");
    }
    if (spec.rotor_momentum <= 0.0) {
      throw std::runtime_error("CMG '" + spec.body_name + "' rotor_momentum must be positive");
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
  std::string xml = read_file(xml_path);
  ParsedOrbitXml parsed = parse_mjorbit_block(&xml);
  if (parsed.plugin_body.empty()) {
    parsed.plugin_body = first_body_name(xml);
  }
  ensure_extension_plugin(&xml);
  ensure_plugin_host(&xml, parsed.plugin_body, parsed.use_j2);

  const std::string temp_path = write_temp_xml(xml);
  char error[2048] = {0};
  mjModel* raw_model = mj_loadXML(temp_path.c_str(), nullptr, error, sizeof(error));
  std::error_code ec;
  std::filesystem::remove(temp_path, ec);
  if (!raw_model) {
    throw std::runtime_error(std::string("MuJoCo XML compile failed: ") + error);
  }

  std::unique_ptr<MjoModel> model(new MjoModel());
  model->model_ = raw_model;
  if (mj_timestep.has_value()) {
    model->model_->opt.timestep = *mj_timestep;
  }
  model->model_->opt.gravity[0] = 0.0;
  model->model_->opt.gravity[1] = 0.0;
  model->model_->opt.gravity[2] = 0.0;
  model->use_j2_ = parsed.use_j2;
  model->use_drag_ = parsed.use_drag;
  model->use_srp_ = parsed.use_srp;
  model->use_magnetic_ = parsed.use_magnetic;
  model->use_gravity_gradient_ = parsed.use_gravity_gradient;
  model->orbit_dt_ = parsed.orbit_dt;

  const int plugin_body_id = resolve_body_id(model->model_, parsed.plugin_body);
  model->orbit_plugin_instance_ = model->model_->body_plugin[plugin_body_id];
  if (model->orbit_plugin_instance_ < 0) {
    throw std::runtime_error("Failed to attach mujoco_orbit plugin to body '" + parsed.plugin_body + "'");
  }

  resolve_config(parsed, model.get());
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

}  // namespace mujoco_orbit
