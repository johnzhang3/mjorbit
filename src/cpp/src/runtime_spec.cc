#include "mujoco_orbit/spec.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "mujoco_orbit/runtime.h"

namespace mujoco_orbit {
namespace {

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

struct RawElement {
  std::string tag;
  std::unordered_map<std::string, std::string> attrs;
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

std::unordered_map<std::string, std::string> parse_attrs(const std::string& text) {
  std::unordered_map<std::string, std::string> attrs;
  const std::regex attr_re(R"ATTR(([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*"([^"]*)")ATTR");
  for (std::sregex_iterator it(text.begin(), text.end(), attr_re), end; it != end; ++it) {
    attrs[(*it)[1].str()] = (*it)[2].str();
  }
  return attrs;
}

bool parse_bool(
    const std::unordered_map<std::string, std::string>& attrs,
    const std::string& key,
    bool default_value) {
  auto it = attrs.find(key);
  if (it == attrs.end()) {
    return default_value;
  }
  const std::string& value = it->second;
  return !(value == "false" || value == "False" || value == "0" || value == "no");
}

double parse_double(
    const std::unordered_map<std::string, std::string>& attrs,
    const std::string& key,
    double default_value) {
  auto it = attrs.find(key);
  if (it == attrs.end() || it->second.empty()) {
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

std::string parse_optional_string(
    const std::unordered_map<std::string, std::string>& attrs,
    const std::string& key) {
  auto it = attrs.find(key);
  return it == attrs.end() ? "" : it->second;
}

void parse_vec3(const std::string& text, std::array<double, 3>* out, const std::string& label) {
  std::istringstream input(text);
  if (!(input >> (*out)[0] >> (*out)[1] >> (*out)[2])) {
    throw std::runtime_error(label + " must contain three numeric values");
  }
  double extra = 0.0;
  if (input >> extra) {
    throw std::runtime_error(label + " must contain exactly three numeric values");
  }
}

void parse_required_vec3(
    const std::unordered_map<std::string, std::string>& attrs,
    const std::string& key,
    const std::string& tag,
    std::array<double, 3>* out) {
  auto it = attrs.find(key);
  if (it == attrs.end()) {
    throw std::runtime_error("<" + tag + "> requires attribute '" + key + "'");
  }
  parse_vec3(it->second, out, "<" + tag + "> " + key);
}

void parse_optional_vec3(
    const std::unordered_map<std::string, std::string>& attrs,
    const std::string& key,
    const std::string& tag,
    std::array<double, 3>* out) {
  auto it = attrs.find(key);
  if (it != attrs.end()) {
    parse_vec3(it->second, out, "<" + tag + "> " + key);
  }
}

std::vector<RawElement> parse_mjorbit_children(const std::string& body) {
  std::vector<RawElement> children;
  const std::regex child_re(R"(<([A-Za-z_][A-Za-z0-9_\-]*)\b([^>]*)/>)");
  for (std::sregex_iterator it(body.begin(), body.end(), child_re), end; it != end; ++it) {
    children.push_back(RawElement{(*it)[1].str(), parse_attrs((*it)[2].str())});
  }
  return children;
}

template <typename T>
bool contains_name(const std::vector<T>& values, const std::string& name) {
  return std::any_of(values.begin(), values.end(), [&](const T& value) {
    return value.name == name;
  });
}

template <typename T>
std::string assign_name(std::vector<T>* values, T* spec, const std::string& prefix) {
  if (spec->name.empty()) {
    for (std::size_t i = values->size();; ++i) {
      const std::string candidate = prefix + "_" + std::to_string(i);
      if (!contains_name(*values, candidate)) {
        spec->name = candidate;
        break;
      }
    }
  }
  if (contains_name(*values, spec->name)) {
    throw std::runtime_error("Duplicate <mjorbit> element name '" + spec->name + "'");
  }
  return spec->name;
}

template <typename T>
std::string add_named(std::vector<T>* values, T spec, const std::string& prefix) {
  const std::string name = assign_name(values, &spec, prefix);
  values->push_back(std::move(spec));
  return name;
}

template <typename T>
void set_named(std::vector<T>* values, const std::string& name, T spec) {
  auto it = std::find_if(values->begin(), values->end(), [&](const T& value) {
    return value.name == name;
  });
  if (it == values->end()) {
    throw std::runtime_error("<mjorbit> element '" + name + "' was not found");
  }
  if (spec.name.empty()) {
    spec.name = name;
  }
  if (spec.name != name && contains_name(*values, spec.name)) {
    throw std::runtime_error("Duplicate <mjorbit> element name '" + spec.name + "'");
  }
  *it = std::move(spec);
}

template <typename T>
void remove_named(std::vector<T>* values, const std::string& name) {
  auto it = std::find_if(values->begin(), values->end(), [&](const T& value) {
    return value.name == name;
  });
  if (it == values->end()) {
    throw std::runtime_error("<mjorbit> element '" + name + "' was not found");
  }
  values->erase(it);
}

double norm3(const std::array<double, 3>& value) {
  return std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
}

void require_finite(double value, const std::string& label) {
  if (!std::isfinite(value)) {
    throw std::runtime_error(label + " must be finite");
  }
}

void validate_central_body(const CentralBodySpecNative& central) {
  require_finite(central.gm, "<central_body> gm");
  require_finite(central.radius, "<central_body> radius");
  require_finite(central.j2, "<central_body> j2");
  require_finite(central.magnetic_b0, "<central_body> magnetic_b0");
  require_finite(central.atmosphere_h0, "<central_body> atmosphere_h0");
  require_finite(central.atmosphere_rho0, "<central_body> atmosphere_rho0");
  require_finite(central.atmosphere_scale_height, "<central_body> atmosphere_scale_height");
  if (central.gm <= 0.0) {
    throw std::runtime_error("<central_body> gm must be positive");
  }
  if (central.radius <= 0.0) {
    throw std::runtime_error("<central_body> radius must be positive");
  }
  if (central.atmosphere_rho0 < 0.0) {
    throw std::runtime_error("<central_body> atmosphere_rho0 must be non-negative");
  }
  if (central.atmosphere_rho0 > 0.0 && central.atmosphere_scale_height <= 0.0) {
    throw std::runtime_error(
        "<central_body> atmosphere_scale_height must be positive when atmosphere_rho0 is positive");
  }
  if (norm3(central.magnetic_axis) < 1.0e-12) {
    throw std::runtime_error("<central_body> magnetic_axis must be non-zero");
  }
  for (int i = 0; i < 3; ++i) {
    require_finite(central.omega[i], "<central_body> omega");
    require_finite(central.magnetic_axis[i], "<central_body> magnetic_axis");
  }
}

OrbitSpecNative parse_mjorbit_block(std::string* xml) {
  OrbitSpecNative parsed;
  const std::regex block_re(R"(<mjorbit\b([^>]*)>([\s\S]*?)</mjorbit>)");
  std::smatch match;
  if (!std::regex_search(*xml, match, block_re)) {
    return parsed;
  }

  const auto attrs = parse_attrs(match[1].str());
  auto plugin_body_it = attrs.find("plugin_body");
  if (plugin_body_it != attrs.end() && !plugin_body_it->second.empty()) {
    parsed.plugin_body = plugin_body_it->second;
  }
  parsed.use_j2 = parse_bool(attrs, "use_j2", true);
  parsed.use_drag = parse_bool(attrs, "use_drag", true);
  parsed.use_srp = parse_bool(attrs, "use_srp", true);
  parsed.use_magnetic = parse_bool(attrs, "use_magnetic", true);
  parsed.use_gravity_gradient = parse_bool(attrs, "use_gravity_gradient", true);
  parsed.orbit_dt = parse_optional_double(attrs, "orbit_dt");

  for (const RawElement& child : parse_mjorbit_children(match[2].str())) {
    if (child.tag == "central_body") {
      CentralBodySpecNative central = parsed.central_body;
      central.name = parse_optional_string(child.attrs, "name");
      if (central.name.empty()) {
        central.name = "earth";
      }
      central.gm = parse_double(child.attrs, "gm", central.gm);
      central.radius = parse_double(child.attrs, "radius", central.radius);
      central.j2 = parse_double(child.attrs, "j2", central.j2);
      parse_optional_vec3(child.attrs, "omega", child.tag, &central.omega);
      central.magnetic_b0 = parse_double(child.attrs, "magnetic_b0", central.magnetic_b0);
      parse_optional_vec3(child.attrs, "magnetic_axis", child.tag, &central.magnetic_axis);
      central.atmosphere_h0 = parse_double(child.attrs, "atmosphere_h0", central.atmosphere_h0);
      central.atmosphere_rho0 =
          parse_double(child.attrs, "atmosphere_rho0", central.atmosphere_rho0);
      central.atmosphere_scale_height = parse_double(
          child.attrs, "atmosphere_scale_height", central.atmosphere_scale_height);
      parsed.central_body = central;
    } else if (child.tag == "surface") {
      OrbitSurfaceSpecNative spec;
      spec.name = parse_optional_string(child.attrs, "name");
      spec.body_name = parse_required_string(child.attrs, "body", child.tag);
      parse_required_vec3(child.attrs, "cop", child.tag, &spec.center_of_pressure_body);
      parse_required_vec3(child.attrs, "normal", child.tag, &spec.normal_body);
      spec.area = parse_double(child.attrs, "area", spec.area);
      spec.drag_coeff = parse_double(child.attrs, "drag_coeff", spec.drag_coeff);
      spec.srp_coeff = parse_double(child.attrs, "srp_coeff", spec.srp_coeff);
      spec.use_drag = parse_bool(child.attrs, "use_drag", true);
      spec.use_srp = parse_bool(child.attrs, "use_srp", true);
      add_named(&parsed.surfaces, std::move(spec), "surface");
    } else if (child.tag == "magnetic_body") {
      OrbitMagneticBodySpecNative spec;
      spec.name = parse_optional_string(child.attrs, "name");
      spec.body_name = parse_required_string(child.attrs, "body", child.tag);
      parse_required_vec3(child.attrs, "dipole", child.tag, &spec.dipole_body);
      add_named(&parsed.magnetic_bodies, std::move(spec), "magnetic_body");
    } else if (child.tag == "reaction_wheel") {
      OrbitReactionWheelSpecNative spec;
      spec.name = parse_optional_string(child.attrs, "name");
      spec.body_name = parse_required_string(child.attrs, "body", child.tag);
      parse_required_vec3(child.attrs, "axis", child.tag, &spec.axis_body);
      spec.inertia = parse_double(child.attrs, "inertia", spec.inertia);
      spec.speed_limit = parse_optional_double(child.attrs, "speed_limit");
      spec.torque_limit = parse_optional_double(child.attrs, "torque_limit");
      add_named(&parsed.reaction_wheels, std::move(spec), "reaction_wheel");
    } else if (child.tag == "magnetorquer") {
      OrbitMagnetorquerSpecNative spec;
      spec.name = parse_optional_string(child.attrs, "name");
      spec.body_name = parse_required_string(child.attrs, "body", child.tag);
      parse_required_vec3(child.attrs, "axis", child.tag, &spec.axis_body);
      spec.dipole_limit = parse_double(child.attrs, "dipole_limit", spec.dipole_limit);
      add_named(&parsed.magnetorquers, std::move(spec), "magnetorquer");
    } else if (child.tag == "thruster") {
      OrbitThrusterSpecNative spec;
      spec.name = parse_optional_string(child.attrs, "name");
      spec.body_name = parse_required_string(child.attrs, "body", child.tag);
      parse_required_vec3(child.attrs, "pos", child.tag, &spec.position_body);
      parse_required_vec3(child.attrs, "dir", child.tag, &spec.direction_body);
      spec.force_limit = parse_double(child.attrs, "force_limit", spec.force_limit);
      add_named(&parsed.thrusters, std::move(spec), "thruster");
    } else if (child.tag == "cmg") {
      OrbitCmgSpecNative spec;
      spec.name = parse_optional_string(child.attrs, "name");
      spec.body_name = parse_required_string(child.attrs, "body", child.tag);
      parse_required_vec3(child.attrs, "gimbal_axis", child.tag, &spec.gimbal_axis_body);
      parse_required_vec3(child.attrs, "spin_axis0", child.tag, &spec.spin_axis_body_0);
      spec.rotor_momentum = parse_double(child.attrs, "rotor_momentum", spec.rotor_momentum);
      spec.gimbal_rate_limit = parse_optional_double(child.attrs, "gimbal_rate_limit");
      spec.gimbal_angle_limit = parse_optional_double(child.attrs, "gimbal_angle_limit");
      add_named(&parsed.cmgs, std::move(spec), "cmg");
    }
  }

  xml->erase(static_cast<std::size_t>(match.position(0)), static_cast<std::size_t>(match.length(0)));
  return parsed;
}

std::string xml_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    switch (ch) {
      case '&': out += "&amp;"; break;
      case '"': out += "&quot;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      default: out.push_back(ch); break;
    }
  }
  return out;
}

std::string format_double(double value) {
  std::ostringstream out;
  out.precision(17);
  out << value;
  return out.str();
}

std::string format_vec3(const std::array<double, 3>& value) {
  return format_double(value[0]) + " " + format_double(value[1]) + " " + format_double(value[2]);
}

void attr(std::ostringstream* out, const std::string& key, const std::string& value) {
  *out << " " << key << "=\"" << xml_escape(value) << "\"";
}

void attr_double(std::ostringstream* out, const std::string& key, double value) {
  attr(out, key, format_double(value));
}

void attr_vec3(std::ostringstream* out, const std::string& key, const std::array<double, 3>& value) {
  attr(out, key, format_vec3(value));
}

void attr_bool(std::ostringstream* out, const std::string& key, bool value) {
  attr(out, key, value ? "true" : "false");
}

std::string serialize_mjorbit(const OrbitSpecNative& orbit) {
  std::ostringstream out;
  out << "  <mjorbit";
  if (orbit.plugin_body.has_value() && !orbit.plugin_body->empty()) {
    attr(&out, "plugin_body", *orbit.plugin_body);
  }
  attr_bool(&out, "use_j2", orbit.use_j2);
  attr_bool(&out, "use_drag", orbit.use_drag);
  attr_bool(&out, "use_srp", orbit.use_srp);
  attr_bool(&out, "use_magnetic", orbit.use_magnetic);
  attr_bool(&out, "use_gravity_gradient", orbit.use_gravity_gradient);
  if (orbit.orbit_dt.has_value()) {
    attr_double(&out, "orbit_dt", *orbit.orbit_dt);
  }
  out << ">\n";

  out << "    <central_body";
  attr(&out, "name", orbit.central_body.name);
  attr_double(&out, "gm", orbit.central_body.gm);
  attr_double(&out, "radius", orbit.central_body.radius);
  attr_double(&out, "j2", orbit.central_body.j2);
  attr_vec3(&out, "omega", orbit.central_body.omega);
  attr_double(&out, "magnetic_b0", orbit.central_body.magnetic_b0);
  attr_vec3(&out, "magnetic_axis", orbit.central_body.magnetic_axis);
  attr_double(&out, "atmosphere_h0", orbit.central_body.atmosphere_h0);
  attr_double(&out, "atmosphere_rho0", orbit.central_body.atmosphere_rho0);
  attr_double(&out, "atmosphere_scale_height", orbit.central_body.atmosphere_scale_height);
  out << "/>\n";

  for (const auto& spec : orbit.surfaces) {
    out << "    <surface";
    attr(&out, "name", spec.name);
    attr(&out, "body", spec.body_name);
    attr_vec3(&out, "cop", spec.center_of_pressure_body);
    attr_vec3(&out, "normal", spec.normal_body);
    attr_double(&out, "area", spec.area);
    attr_double(&out, "drag_coeff", spec.drag_coeff);
    attr_double(&out, "srp_coeff", spec.srp_coeff);
    attr_bool(&out, "use_drag", spec.use_drag);
    attr_bool(&out, "use_srp", spec.use_srp);
    out << "/>\n";
  }
  for (const auto& spec : orbit.magnetic_bodies) {
    out << "    <magnetic_body";
    attr(&out, "name", spec.name);
    attr(&out, "body", spec.body_name);
    attr_vec3(&out, "dipole", spec.dipole_body);
    out << "/>\n";
  }
  for (const auto& spec : orbit.reaction_wheels) {
    out << "    <reaction_wheel";
    attr(&out, "name", spec.name);
    attr(&out, "body", spec.body_name);
    attr_vec3(&out, "axis", spec.axis_body);
    attr_double(&out, "inertia", spec.inertia);
    if (spec.speed_limit.has_value()) attr_double(&out, "speed_limit", *spec.speed_limit);
    if (spec.torque_limit.has_value()) attr_double(&out, "torque_limit", *spec.torque_limit);
    out << "/>\n";
  }
  for (const auto& spec : orbit.magnetorquers) {
    out << "    <magnetorquer";
    attr(&out, "name", spec.name);
    attr(&out, "body", spec.body_name);
    attr_vec3(&out, "axis", spec.axis_body);
    attr_double(&out, "dipole_limit", spec.dipole_limit);
    out << "/>\n";
  }
  for (const auto& spec : orbit.thrusters) {
    out << "    <thruster";
    attr(&out, "name", spec.name);
    attr(&out, "body", spec.body_name);
    attr_vec3(&out, "pos", spec.position_body);
    attr_vec3(&out, "dir", spec.direction_body);
    attr_double(&out, "force_limit", spec.force_limit);
    out << "/>\n";
  }
  for (const auto& spec : orbit.cmgs) {
    out << "    <cmg";
    attr(&out, "name", spec.name);
    attr(&out, "body", spec.body_name);
    attr_vec3(&out, "gimbal_axis", spec.gimbal_axis_body);
    attr_vec3(&out, "spin_axis0", spec.spin_axis_body_0);
    attr_double(&out, "rotor_momentum", spec.rotor_momentum);
    if (spec.gimbal_rate_limit.has_value()) {
      attr_double(&out, "gimbal_rate_limit", *spec.gimbal_rate_limit);
    }
    if (spec.gimbal_angle_limit.has_value()) {
      attr_double(&out, "gimbal_angle_limit", *spec.gimbal_angle_limit);
    }
    out << "/>\n";
  }

  out << "  </mjorbit>\n";
  return out.str();
}

std::string insert_mjorbit_block(std::string xml, const OrbitSpecNative& orbit) {
  const std::size_t mujoco_end = xml.rfind("</mujoco>");
  if (mujoco_end == std::string::npos) {
    throw std::runtime_error("Serialized MuJoCo XML did not contain </mujoco>");
  }
  xml.insert(mujoco_end, serialize_mjorbit(orbit));
  return xml;
}

struct ParsedSpecXml {
  mjSpec* spec = nullptr;
  std::string xml;
  OrbitSpecNative orbit;
  AssetMap assets;
};

ParsedSpecXml parse_spec_xml(std::string xml, AssetMap assets) {
  ParsedSpecXml out;
  out.orbit = parse_mjorbit_block(&xml);
  out.xml = xml;
  out.assets = std::move(assets);

  char error[2048] = {0};
  VfsHolder vfs(out.assets);
  out.spec = mj_parseXMLString(xml.c_str(), vfs.get(), error, sizeof(error));
  if (!out.spec) {
    throw std::runtime_error(std::string("MuJoCo XML parse failed: ") + error);
  }
  return out;
}

}  // namespace

MjoSpec::~MjoSpec() {
  if (spec_) {
    mj_deleteSpec(spec_);
    spec_ = nullptr;
  }
}

std::unique_ptr<MjoSpec> MjoSpec::FromXmlPath(const std::string& xml_path) {
  return FromXmlString(read_file(xml_path), {});
}

std::unique_ptr<MjoSpec> MjoSpec::FromXmlString(const std::string& xml, AssetMap assets) {
  ParsedSpecXml parsed = parse_spec_xml(xml, std::move(assets));
  std::unique_ptr<MjoSpec> out(new MjoSpec());
  out->spec_ = parsed.spec;
  out->xml_ = std::move(parsed.xml);
  out->orbit_ = std::move(parsed.orbit);
  out->assets_ = std::move(parsed.assets);
  return out;
}

std::unique_ptr<MjoSpec> MjoSpec::Copy() const {
  std::unique_ptr<MjoSpec> out(new MjoSpec());
  out->spec_ = mj_copySpec(spec_);
  if (!out->spec_) {
    throw std::runtime_error("mj_copySpec failed");
  }
  out->xml_ = xml_;
  out->orbit_ = orbit_;
  out->assets_ = assets_;
  return out;
}

std::unique_ptr<MjoModel> MjoSpec::Compile(std::optional<double> mj_timestep) const {
  validate_central_body(orbit_.central_body);
  return MjoModel::FromSpecXml(xml_, orbit_, assets_, mj_timestep);
}

std::string MjoSpec::ToXml() const {
  return insert_mjorbit_block(xml_, orbit_);
}

std::string MjoSpec::AddSurface(OrbitSurfaceSpecNative spec) {
  return add_named(&orbit_.surfaces, std::move(spec), "surface");
}

void MjoSpec::SetSurface(const std::string& name, OrbitSurfaceSpecNative spec) {
  set_named(&orbit_.surfaces, name, std::move(spec));
}

void MjoSpec::RemoveSurface(const std::string& name) {
  remove_named(&orbit_.surfaces, name);
}

std::string MjoSpec::AddMagneticBody(OrbitMagneticBodySpecNative spec) {
  return add_named(&orbit_.magnetic_bodies, std::move(spec), "magnetic_body");
}

void MjoSpec::SetMagneticBody(const std::string& name, OrbitMagneticBodySpecNative spec) {
  set_named(&orbit_.magnetic_bodies, name, std::move(spec));
}

void MjoSpec::RemoveMagneticBody(const std::string& name) {
  remove_named(&orbit_.magnetic_bodies, name);
}

std::string MjoSpec::AddReactionWheel(OrbitReactionWheelSpecNative spec) {
  return add_named(&orbit_.reaction_wheels, std::move(spec), "reaction_wheel");
}

void MjoSpec::SetReactionWheel(const std::string& name, OrbitReactionWheelSpecNative spec) {
  set_named(&orbit_.reaction_wheels, name, std::move(spec));
}

void MjoSpec::RemoveReactionWheel(const std::string& name) {
  remove_named(&orbit_.reaction_wheels, name);
}

std::string MjoSpec::AddMagnetorquer(OrbitMagnetorquerSpecNative spec) {
  return add_named(&orbit_.magnetorquers, std::move(spec), "magnetorquer");
}

void MjoSpec::SetMagnetorquer(const std::string& name, OrbitMagnetorquerSpecNative spec) {
  set_named(&orbit_.magnetorquers, name, std::move(spec));
}

void MjoSpec::RemoveMagnetorquer(const std::string& name) {
  remove_named(&orbit_.magnetorquers, name);
}

std::string MjoSpec::AddThruster(OrbitThrusterSpecNative spec) {
  return add_named(&orbit_.thrusters, std::move(spec), "thruster");
}

void MjoSpec::SetThruster(const std::string& name, OrbitThrusterSpecNative spec) {
  set_named(&orbit_.thrusters, name, std::move(spec));
}

void MjoSpec::RemoveThruster(const std::string& name) {
  remove_named(&orbit_.thrusters, name);
}

std::string MjoSpec::AddCmg(OrbitCmgSpecNative spec) {
  return add_named(&orbit_.cmgs, std::move(spec), "cmg");
}

void MjoSpec::SetCmg(const std::string& name, OrbitCmgSpecNative spec) {
  set_named(&orbit_.cmgs, name, std::move(spec));
}

void MjoSpec::RemoveCmg(const std::string& name) {
  remove_named(&orbit_.cmgs, name);
}

}  // namespace mujoco_orbit
