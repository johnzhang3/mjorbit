#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "mujoco_orbit/runtime.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace mujoco_orbit {
namespace {

template <typename T>
nb::ndarray<nb::numpy, T> view(T* ptr, std::initializer_list<size_t> shape) {
  return nb::ndarray<nb::numpy, T>(ptr, shape);
}

template <typename T>
nb::ndarray<nb::numpy, const T> const_view(
    const T* ptr,
    std::initializer_list<size_t> shape) {
  return nb::ndarray<nb::numpy, const T>(ptr, shape);
}

void copy_vec3(const std::array<double, 3>& src, double* dst) {
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
}

std::string name_or_default(
    const mjModel* model,
    mjtObj object_type,
    int object_id,
    const std::string& fallback_prefix) {
  const char* name = mj_id2name(model, object_type, object_id);
  if (name != nullptr) {
    return std::string(name);
  }
  return fallback_prefix + "_" + std::to_string(object_id);
}

struct ModelOptView {
  MjoModel* model;
};

struct OrbitView {
  MjoData* data;
};

struct FrameView {
  MjoData* data;
};

struct EnvironmentView {
  MjoData* data;
};

struct ActuatorView {
  MjoData* data;
};

struct SensorDataView {
  MjoData* data;
};

std::vector<double> get_state_vector(const MjoModel& model, const MjoData& data) {
  std::vector<double> state(static_cast<std::size_t>(mjo_state_size(model)), 0.0);
  mjo_get_state(model, data, state.data());
  return state;
}

void set_state_array(
    const MjoModel& model,
    MjoData& data,
    nb::ndarray<nb::numpy, const double, nb::c_contig> state) {
  mjo_set_state(model, data, state.data(), static_cast<int>(state.size()));
}

int rollout_array(
    MjoModel& model,
    MjoData& data,
    int nbatch,
    int nstep,
    unsigned int control_spec,
    int state_size,
    int control_size,
    nb::object initial_state,
    nb::object initial_warmstart,
    nb::object control,
    nb::object state,
    nb::object sensordata) {
  const double* initial_warmstart_ptr = nullptr;
  const double* control_ptr = nullptr;
  double* state_ptr = nullptr;
  double* sensordata_ptr = nullptr;

  auto initial_state_arr = nb::cast<nb::ndarray<nb::numpy, double, nb::c_contig>>(initial_state);
  nb::ndarray<nb::numpy, double, nb::c_contig> initial_warmstart_arr;
  nb::ndarray<nb::numpy, double, nb::c_contig> control_arr;
  nb::ndarray<nb::numpy, double, nb::c_contig> state_arr;
  nb::ndarray<nb::numpy, double, nb::c_contig> sensordata_arr;
  initial_warmstart_arr =
      nb::cast<nb::ndarray<nb::numpy, double, nb::c_contig>>(initial_warmstart);
  if (initial_warmstart_arr.size() > 0) {
    initial_warmstart_ptr = initial_warmstart_arr.data();
  }
  control_arr = nb::cast<nb::ndarray<nb::numpy, double, nb::c_contig>>(control);
  if (control_arr.size() > 0) {
    control_ptr = control_arr.data();
  }
  state_arr = nb::cast<nb::ndarray<nb::numpy, double, nb::c_contig>>(state);
  if (state_arr.size() > 0) {
    state_ptr = state_arr.data();
  }
  sensordata_arr = nb::cast<nb::ndarray<nb::numpy, double, nb::c_contig>>(sensordata);
  if (sensordata_arr.size() > 0) {
    sensordata_ptr = sensordata_arr.data();
  }

  const double* initial_state_ptr = initial_state_arr.data();
  nb::gil_scoped_release release;
  return mjo_rollout_native(
      model,
      data,
      nbatch,
      nstep,
      control_spec,
      state_size,
      control_size,
      initial_state_ptr,
      initial_warmstart_ptr,
      control_ptr,
      state_ptr,
      sensordata_ptr);
}

std::array<double, 6> contact_force(MjoData& data, int contact_id) {
  if (contact_id < 0 || contact_id >= data.raw()->ncon) {
    throw std::runtime_error("contact_id out of range");
  }
  std::array<double, 6> force = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  mj_contactForce(data.model().raw(), data.raw(), contact_id, force.data());
  return force;
}

std::vector<std::array<double, 6>> contact_force_segments(MjoData& data, double force_scale) {
  std::vector<std::array<double, 6>> segments;
  for (int contact_id = 0; contact_id < data.raw()->ncon; ++contact_id) {
    std::array<double, 6> force = contact_force(data, contact_id);
    const mjContact& contact = data.raw()->contact[contact_id];
    double force_world[3] = {
        contact.frame[0] * force[0] + contact.frame[3] * force[1] +
            contact.frame[6] * force[2],
        contact.frame[1] * force[0] + contact.frame[4] * force[1] +
            contact.frame[7] * force[2],
        contact.frame[2] * force[0] + contact.frame[5] * force[1] +
            contact.frame[8] * force[2],
    };
    const double norm = std::sqrt(
        force_world[0] * force_world[0] +
        force_world[1] * force_world[1] +
        force_world[2] * force_world[2]);
    if (norm < 1.0e-9) {
      continue;
    }
    segments.push_back({
        contact.pos[0],
        contact.pos[1],
        contact.pos[2],
        contact.pos[0] + force_scale * force_world[0],
        contact.pos[1] + force_scale * force_world[1],
        contact.pos[2] + force_scale * force_world[2],
    });
  }
  return segments;
}

}  // namespace
}  // namespace mujoco_orbit

NB_MODULE(_bindings, m) {
  using namespace mujoco_orbit;

  nb::class_<SensorDescriptor>(m, "SensorDescriptor")
      .def_ro("sensor_id", &SensorDescriptor::sensor_id)
      .def_ro("name", &SensorDescriptor::name)
      .def_ro("sensor_type", &SensorDescriptor::sensor_type)
      .def_ro("datatype", &SensorDescriptor::datatype)
      .def_ro("objtype", &SensorDescriptor::objtype)
      .def_ro("objid", &SensorDescriptor::objid)
      .def_ro("adr", &SensorDescriptor::adr)
      .def_ro("dim", &SensorDescriptor::dim)
      .def_ro("noise", &SensorDescriptor::noise)
      .def_ro("cutoff", &SensorDescriptor::cutoff)
      .def_ro("needstage", &SensorDescriptor::needstage)
      .def_ro("user", &SensorDescriptor::user)
      .def_ro("orbit_kind", &SensorDescriptor::orbit_kind)
      .def_ro("reference_eci", &SensorDescriptor::reference_eci)
      .def_prop_ro("data_slice", [](const SensorDescriptor& self) {
        return nb::slice(self.adr, self.adr + self.dim, 1);
      });

  nb::class_<SensorCatalog>(m, "SensorCatalog")
      .def_ro("descriptors", &SensorCatalog::descriptors)
      .def_ro("by_name", &SensorCatalog::by_name)
      .def_ro("custom_descriptors", &SensorCatalog::custom_descriptors);

  nb::class_<CentralBodySpecNative>(m, "CentralBodySpec")
      .def(nb::init<>())
      .def_rw("name", &CentralBodySpecNative::name)
      .def_rw("gm", &CentralBodySpecNative::gm)
      .def_rw("radius", &CentralBodySpecNative::radius)
      .def_rw("j2", &CentralBodySpecNative::j2)
      .def_rw("omega", &CentralBodySpecNative::omega)
      .def_rw("magnetic_b0", &CentralBodySpecNative::magnetic_b0)
      .def_rw("magnetic_axis", &CentralBodySpecNative::magnetic_axis)
      .def_rw("atmosphere_h0", &CentralBodySpecNative::atmosphere_h0)
      .def_rw("atmosphere_rho0", &CentralBodySpecNative::atmosphere_rho0)
      .def_rw("atmosphere_scale_height", &CentralBodySpecNative::atmosphere_scale_height);

  nb::class_<OrbitSurfaceSpecNative>(m, "OrbitSurfaceSpec")
      .def(nb::init<>())
      .def_rw("name", &OrbitSurfaceSpecNative::name)
      .def_rw("body_name", &OrbitSurfaceSpecNative::body_name)
      .def_rw(
          "center_of_pressure_body",
          &OrbitSurfaceSpecNative::center_of_pressure_body)
      .def_rw("normal_body", &OrbitSurfaceSpecNative::normal_body)
      .def_rw("area", &OrbitSurfaceSpecNative::area)
      .def_rw("drag_coeff", &OrbitSurfaceSpecNative::drag_coeff)
      .def_rw("srp_coeff", &OrbitSurfaceSpecNative::srp_coeff)
      .def_rw("use_drag", &OrbitSurfaceSpecNative::use_drag)
      .def_rw("use_srp", &OrbitSurfaceSpecNative::use_srp);

  nb::class_<OrbitMagneticBodySpecNative>(m, "OrbitMagneticBodySpec")
      .def(nb::init<>())
      .def_rw("name", &OrbitMagneticBodySpecNative::name)
      .def_rw("body_name", &OrbitMagneticBodySpecNative::body_name)
      .def_rw("dipole_body", &OrbitMagneticBodySpecNative::dipole_body);

  nb::class_<OrbitReactionWheelSpecNative>(m, "OrbitReactionWheelSpec")
      .def(nb::init<>())
      .def_rw("name", &OrbitReactionWheelSpecNative::name)
      .def_rw("body_name", &OrbitReactionWheelSpecNative::body_name)
      .def_rw("axis_body", &OrbitReactionWheelSpecNative::axis_body)
      .def_rw("inertia", &OrbitReactionWheelSpecNative::inertia)
      .def_rw("speed_limit", &OrbitReactionWheelSpecNative::speed_limit)
      .def_rw("torque_limit", &OrbitReactionWheelSpecNative::torque_limit);

  nb::class_<OrbitMagnetorquerSpecNative>(m, "OrbitMagnetorquerSpec")
      .def(nb::init<>())
      .def_rw("name", &OrbitMagnetorquerSpecNative::name)
      .def_rw("body_name", &OrbitMagnetorquerSpecNative::body_name)
      .def_rw("axis_body", &OrbitMagnetorquerSpecNative::axis_body)
      .def_rw("dipole_limit", &OrbitMagnetorquerSpecNative::dipole_limit);

  nb::class_<OrbitThrusterSpecNative>(m, "OrbitThrusterSpec")
      .def(nb::init<>())
      .def_rw("name", &OrbitThrusterSpecNative::name)
      .def_rw("body_name", &OrbitThrusterSpecNative::body_name)
      .def_rw("position_body", &OrbitThrusterSpecNative::position_body)
      .def_rw("direction_body", &OrbitThrusterSpecNative::direction_body)
      .def_rw("force_limit", &OrbitThrusterSpecNative::force_limit);

  nb::class_<OrbitCmgSpecNative>(m, "OrbitCmgSpec")
      .def(nb::init<>())
      .def_rw("name", &OrbitCmgSpecNative::name)
      .def_rw("body_name", &OrbitCmgSpecNative::body_name)
      .def_rw("gimbal_axis_body", &OrbitCmgSpecNative::gimbal_axis_body)
      .def_rw("spin_axis_body_0", &OrbitCmgSpecNative::spin_axis_body_0)
      .def_rw("rotor_momentum", &OrbitCmgSpecNative::rotor_momentum)
      .def_rw("gimbal_rate_limit", &OrbitCmgSpecNative::gimbal_rate_limit)
      .def_rw("gimbal_angle_limit", &OrbitCmgSpecNative::gimbal_angle_limit);

  nb::class_<SurfaceMetadataNative>(m, "SurfaceMetadata")
      .def_ro("body_id", &SurfaceMetadataNative::body_id)
      .def_prop_ro("center_of_pressure_body", [](SurfaceMetadataNative& self) {
        return view(self.center_of_pressure_body, {static_cast<size_t>(3)});
      })
      .def_prop_ro("normal_body", [](SurfaceMetadataNative& self) {
        return view(self.normal_body, {static_cast<size_t>(3)});
      })
      .def_ro("area", &SurfaceMetadataNative::area)
      .def_ro("drag_coeff", &SurfaceMetadataNative::drag_coeff)
      .def_ro("srp_coeff", &SurfaceMetadataNative::srp_coeff)
      .def_prop_ro("use_drag", [](const SurfaceMetadataNative& self) {
        return self.use_drag != 0;
      })
      .def_prop_ro("use_srp", [](const SurfaceMetadataNative& self) {
        return self.use_srp != 0;
      });

  nb::class_<MagneticMetadataNative>(m, "MagneticMetadata")
      .def_ro("body_id", &MagneticMetadataNative::body_id)
      .def_prop_ro("dipole_body", [](MagneticMetadataNative& self) {
        return view(self.dipole_body, {static_cast<size_t>(3)});
      });

  nb::class_<ReactionWheelMetadataNative>(m, "ReactionWheelMetadata")
      .def_ro("body_id", &ReactionWheelMetadataNative::body_id)
      .def_prop_ro("axis_body", [](ReactionWheelMetadataNative& self) {
        return view(self.axis_body, {static_cast<size_t>(3)});
      })
      .def_ro("inertia", &ReactionWheelMetadataNative::inertia)
      .def_ro("speed_limit", &ReactionWheelMetadataNative::speed_limit)
      .def_ro("torque_limit", &ReactionWheelMetadataNative::torque_limit)
      .def_prop_ro("has_speed_limit", [](const ReactionWheelMetadataNative& self) {
        return self.has_speed_limit != 0;
      })
      .def_prop_ro("has_torque_limit", [](const ReactionWheelMetadataNative& self) {
        return self.has_torque_limit != 0;
      });

  nb::class_<MagnetorquerMetadataNative>(m, "MagnetorquerMetadata")
      .def_ro("body_id", &MagnetorquerMetadataNative::body_id)
      .def_prop_ro("axis_body", [](MagnetorquerMetadataNative& self) {
        return view(self.axis_body, {static_cast<size_t>(3)});
      })
      .def_ro("dipole_limit", &MagnetorquerMetadataNative::dipole_limit);

  nb::class_<ThrusterMetadataNative>(m, "ThrusterMetadata")
      .def_ro("body_id", &ThrusterMetadataNative::body_id)
      .def_prop_ro("position_body", [](ThrusterMetadataNative& self) {
        return view(self.position_body, {static_cast<size_t>(3)});
      })
      .def_prop_ro("direction_body", [](ThrusterMetadataNative& self) {
        return view(self.direction_body, {static_cast<size_t>(3)});
      })
      .def_ro("force_limit", &ThrusterMetadataNative::force_limit);

  nb::class_<ControlMomentGyroMetadataNative>(m, "ControlMomentGyroMetadata")
      .def_ro("body_id", &ControlMomentGyroMetadataNative::body_id)
      .def_prop_ro("gimbal_axis_body", [](ControlMomentGyroMetadataNative& self) {
        return view(self.gimbal_axis_body, {static_cast<size_t>(3)});
      })
      .def_prop_ro("spin_axis_body_0", [](ControlMomentGyroMetadataNative& self) {
        return view(self.spin_axis_body_0, {static_cast<size_t>(3)});
      })
      .def_prop_ro("torque_axis_body_0", [](ControlMomentGyroMetadataNative& self) {
        return view(self.torque_axis_body_0, {static_cast<size_t>(3)});
      })
      .def_ro("rotor_momentum", &ControlMomentGyroMetadataNative::rotor_momentum)
      .def_ro("gimbal_rate_limit", &ControlMomentGyroMetadataNative::gimbal_rate_limit)
      .def_ro("gimbal_angle_limit", &ControlMomentGyroMetadataNative::gimbal_angle_limit)
      .def_prop_ro("has_gimbal_rate_limit", [](const ControlMomentGyroMetadataNative& self) {
        return self.has_gimbal_rate_limit != 0;
      })
      .def_prop_ro("has_gimbal_angle_limit", [](const ControlMomentGyroMetadataNative& self) {
        return self.has_gimbal_angle_limit != 0;
      });

  nb::class_<MjoSpec>(m, "MjoSpec")
      .def_static("from_xml_path", &MjoSpec::FromXmlPath, "xml_path"_a)
      .def_static(
          "from_xml_string",
          &MjoSpec::FromXmlString,
          "xml"_a,
          "assets"_a = AssetMap{})
      .def("copy", &MjoSpec::Copy)
      .def("compile", &MjoSpec::Compile, "mj_timestep"_a = nb::none())
      .def("to_xml", &MjoSpec::ToXml)
      .def_prop_rw("plugin_body", &MjoSpec::plugin_body, &MjoSpec::set_plugin_body)
      .def_prop_rw("use_j2", &MjoSpec::use_j2, &MjoSpec::set_use_j2)
      .def_prop_rw("use_drag", &MjoSpec::use_drag, &MjoSpec::set_use_drag)
      .def_prop_rw("use_srp", &MjoSpec::use_srp, &MjoSpec::set_use_srp)
      .def_prop_rw("use_magnetic", &MjoSpec::use_magnetic, &MjoSpec::set_use_magnetic)
      .def_prop_rw(
          "use_gravity_gradient",
          &MjoSpec::use_gravity_gradient,
          &MjoSpec::set_use_gravity_gradient)
      .def_prop_rw("orbit_dt", &MjoSpec::orbit_dt, &MjoSpec::set_orbit_dt)
      .def_prop_rw(
          "central_body",
          [](MjoSpec& self) -> CentralBodySpecNative& {
            return self.orbit().central_body;
          },
          [](MjoSpec& self, const CentralBodySpecNative& value) {
            self.set_central_body(value);
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("surfaces", [](MjoSpec& self) -> const std::vector<OrbitSurfaceSpecNative>& {
        return self.orbit().surfaces;
      }, nb::rv_policy::reference_internal)
      .def_prop_ro(
          "magnetic_bodies",
          [](MjoSpec& self) -> const std::vector<OrbitMagneticBodySpecNative>& {
            return self.orbit().magnetic_bodies;
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "reaction_wheels",
          [](MjoSpec& self) -> const std::vector<OrbitReactionWheelSpecNative>& {
            return self.orbit().reaction_wheels;
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "magnetorquers",
          [](MjoSpec& self) -> const std::vector<OrbitMagnetorquerSpecNative>& {
            return self.orbit().magnetorquers;
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("thrusters", [](MjoSpec& self) -> const std::vector<OrbitThrusterSpecNative>& {
        return self.orbit().thrusters;
      }, nb::rv_policy::reference_internal)
      .def_prop_ro("cmgs", [](MjoSpec& self) -> const std::vector<OrbitCmgSpecNative>& {
        return self.orbit().cmgs;
      }, nb::rv_policy::reference_internal)
      .def("add_surface", &MjoSpec::AddSurface, "spec"_a)
      .def("update_surface", &MjoSpec::SetSurface, "name"_a, "spec"_a)
      .def("remove_surface", &MjoSpec::RemoveSurface, "name"_a)
      .def("add_magnetic_body", &MjoSpec::AddMagneticBody, "spec"_a)
      .def("update_magnetic_body", &MjoSpec::SetMagneticBody, "name"_a, "spec"_a)
      .def("remove_magnetic_body", &MjoSpec::RemoveMagneticBody, "name"_a)
      .def("add_reaction_wheel", &MjoSpec::AddReactionWheel, "spec"_a)
      .def("update_reaction_wheel", &MjoSpec::SetReactionWheel, "name"_a, "spec"_a)
      .def("remove_reaction_wheel", &MjoSpec::RemoveReactionWheel, "name"_a)
      .def("add_magnetorquer", &MjoSpec::AddMagnetorquer, "spec"_a)
      .def("update_magnetorquer", &MjoSpec::SetMagnetorquer, "name"_a, "spec"_a)
      .def("remove_magnetorquer", &MjoSpec::RemoveMagnetorquer, "name"_a)
      .def("add_thruster", &MjoSpec::AddThruster, "spec"_a)
      .def("update_thruster", &MjoSpec::SetThruster, "name"_a, "spec"_a)
      .def("remove_thruster", &MjoSpec::RemoveThruster, "name"_a)
      .def("add_cmg", &MjoSpec::AddCmg, "spec"_a)
      .def("update_cmg", &MjoSpec::SetCmg, "name"_a, "spec"_a)
      .def("remove_cmg", &MjoSpec::RemoveCmg, "name"_a);

  nb::class_<MjoModel>(m, "MjoModel")
      .def_static("from_xml_path", &MjoModel::FromXmlPath, "xml_path"_a, "mj_timestep"_a = nb::none())
      .def("body_id", &MjoModel::body_id)
      .def("body_name", [](MjoModel& self, int body_id) {
        return name_or_default(self.raw(), mjOBJ_BODY, body_id, "body");
      })
      .def("geom_name", [](MjoModel& self, int geom_id) {
        return name_or_default(self.raw(), mjOBJ_GEOM, geom_id, "geom");
      })
      .def("sensor", &MjoModel::sensor, nb::rv_policy::reference_internal)
      .def_prop_ro("opt", [](MjoModel& self) { return ModelOptView{&self}; })
      .def_prop_ro("sensors", [](MjoModel& self) -> const SensorCatalog& {
        return self.sensors();
      }, nb::rv_policy::reference_internal)
      .def_prop_ro("surfaces", [](MjoModel& self) -> const std::vector<SurfaceMetadataNative>& {
        return self.surfaces();
      }, nb::rv_policy::reference_internal)
      .def_prop_ro("magnetic_bodies", [](MjoModel& self) -> const std::vector<MagneticMetadataNative>& {
        return self.magnetic_bodies();
      }, nb::rv_policy::reference_internal)
      .def_prop_ro("reaction_wheels", [](MjoModel& self) -> const std::vector<ReactionWheelMetadataNative>& {
        return self.reaction_wheels();
      }, nb::rv_policy::reference_internal)
      .def_prop_ro("magnetorquers", [](MjoModel& self) -> const std::vector<MagnetorquerMetadataNative>& {
        return self.magnetorquers();
      }, nb::rv_policy::reference_internal)
      .def_prop_ro("thrusters", [](MjoModel& self) -> const std::vector<ThrusterMetadataNative>& {
        return self.thrusters();
      }, nb::rv_policy::reference_internal)
      .def_prop_ro("cmgs", [](MjoModel& self) -> const std::vector<ControlMomentGyroMetadataNative>& {
        return self.cmgs();
      }, nb::rv_policy::reference_internal)
      .def_prop_ro("orbit_plugin_instance", &MjoModel::orbit_plugin_instance)
      .def_prop_ro("nbody", &MjoModel::nbody)
      .def_prop_ro("nq", &MjoModel::nq)
      .def_prop_ro("nv", &MjoModel::nv)
      .def_prop_ro("nu", &MjoModel::nu)
      .def_prop_ro("na", &MjoModel::na)
      .def_prop_ro("nmocap", &MjoModel::nmocap)
      .def_prop_ro("neq", &MjoModel::neq)
      .def_prop_ro("njnt", [](MjoModel& self) { return self.raw()->njnt; })
      .def_prop_ro("nsensordata", &MjoModel::nsensordata)
      .def_prop_ro("nsensor", &MjoModel::nsensor)
      .def_prop_ro("ngeom", &MjoModel::ngeom)
      .def_prop_ro("nsite", [](MjoModel& self) { return self.raw()->nsite; })
      .def_prop_ro("nplugin", &MjoModel::nplugin)
      .def_prop_ro("use_j2", &MjoModel::use_j2)
      .def_prop_ro("use_drag", &MjoModel::use_drag)
      .def_prop_ro("use_srp", &MjoModel::use_srp)
      .def_prop_ro("use_magnetic", &MjoModel::use_magnetic)
      .def_prop_ro("use_gravity_gradient", &MjoModel::use_gravity_gradient)
      .def_prop_ro("orbit_dt", &MjoModel::orbit_dt)
      .def_prop_ro("central_body", [](const MjoModel& self) {
        return self.central_body();
      })
      .def_prop_ro("body_mass", [](MjoModel& self) {
        return view(self.raw()->body_mass, {static_cast<size_t>(self.nbody())});
      })
      .def_prop_ro("body_inertia", [](MjoModel& self) {
        return view(self.raw()->body_inertia, {static_cast<size_t>(self.nbody()), 3});
      })
      .def_prop_ro("body_ipos", [](MjoModel& self) {
        return view(self.raw()->body_ipos, {static_cast<size_t>(self.nbody()), 3});
      })
      .def_prop_ro("body_iquat", [](MjoModel& self) {
        return view(self.raw()->body_iquat, {static_cast<size_t>(self.nbody()), 4});
      })
      .def_prop_ro("rw_inertia", [](MjoModel& self) {
        std::vector<double> out;
        out.reserve(self.reaction_wheels().size());
        for (const auto& wheel : self.reaction_wheels()) {
          out.push_back(wheel.inertia);
        }
        return out;
      })
      .def_prop_ro("jnt_dofadr", [](MjoModel& self) {
        return view(self.raw()->jnt_dofadr, {static_cast<size_t>(self.raw()->njnt)});
      })
      .def_prop_ro("geom_bodyid", [](MjoModel& self) {
        return view(self.raw()->geom_bodyid, {static_cast<size_t>(self.raw()->ngeom)});
      })
      .def_prop_ro("geom_type", [](MjoModel& self) {
        return view(self.raw()->geom_type, {static_cast<size_t>(self.raw()->ngeom)});
      })
      .def_prop_ro("geom_size", [](MjoModel& self) {
        return view(self.raw()->geom_size, {static_cast<size_t>(self.raw()->ngeom), 3});
      })
      .def_prop_ro("geom_pos", [](MjoModel& self) {
        return view(self.raw()->geom_pos, {static_cast<size_t>(self.raw()->ngeom), 3});
      })
      .def_prop_ro("geom_quat", [](MjoModel& self) {
        return view(self.raw()->geom_quat, {static_cast<size_t>(self.raw()->ngeom), 4});
      })
      .def_prop_ro("geom_rgba", [](MjoModel& self) {
        return view(self.raw()->geom_rgba, {static_cast<size_t>(self.raw()->ngeom), 4});
      });

  nb::class_<ModelOptView>(m, "ModelOptView")
      .def_prop_rw(
          "timestep",
          [](const ModelOptView& self) { return self.model->raw()->opt.timestep; },
          [](ModelOptView& self, double value) { self.model->raw()->opt.timestep = value; })
      .def_prop_ro("gravity", [](ModelOptView& self) {
        return view(self.model->raw()->opt.gravity, {static_cast<size_t>(3)});
      })
      .def_prop_rw(
          "integrator",
          [](const ModelOptView& self) { return static_cast<int>(self.model->raw()->opt.integrator); },
          [](ModelOptView& self, int value) {
            self.model->raw()->opt.integrator = static_cast<mjtIntegrator>(value);
          });

  nb::class_<MjoData>(m, "MjoData")
      .def(nb::init<MjoModel&, const std::array<double, 3>&,
                    const std::array<double, 3>&, double, std::optional<std::uint64_t>>(),
           "model"_a, "R_eci"_a, "V_eci"_a, "t"_a = 0.0, "rng_seed"_a = nb::none(),
           nb::keep_alive<1, 2>())
      .def("reset", &MjoData::reset, "R_eci"_a = nb::none(), "V_eci"_a = nb::none(), "t"_a = nb::none())
      .def("clear_wrench_buffer", &MjoData::clear_wrench_buffer)
      .def("world_position_from_lvlh", &MjoData::world_position_from_lvlh)
      .def("world_velocity_from_lvlh", &MjoData::world_velocity_from_lvlh)
      .def("lvlh_position_from_world", &MjoData::lvlh_position_from_world)
      .def("lvlh_velocity_from_world", &MjoData::lvlh_velocity_from_world)
      .def("contact_force", &contact_force)
      .def("contact_force_segments", &contact_force_segments, "force_scale"_a)
      .def_prop_ro("time", [](MjoData& self) { return self.raw()->time; })
      .def_prop_ro("ncon", [](MjoData& self) { return self.raw()->ncon; })
      .def_prop_ro("qpos", [](MjoData& self) {
        return view(self.raw()->qpos, {static_cast<size_t>(self.model().nq())});
      })
      .def_prop_ro("qvel", [](MjoData& self) {
        return view(self.raw()->qvel, {static_cast<size_t>(self.model().nv())});
      })
      .def_prop_ro("qacc", [](MjoData& self) {
        return view(self.raw()->qacc, {static_cast<size_t>(self.model().nv())});
      })
      .def_prop_ro("act", [](MjoData& self) {
        return view(self.raw()->act, {static_cast<size_t>(self.model().na())});
      })
      .def_prop_ro("ctrl", [](MjoData& self) {
        return view(self.raw()->ctrl, {static_cast<size_t>(self.model().nu())});
      })
      .def_prop_ro("actuator_force", [](MjoData& self) {
        return view(self.raw()->actuator_force, {static_cast<size_t>(self.model().nu())});
      })
      .def_prop_ro("qfrc_actuator", [](MjoData& self) {
        return view(self.raw()->qfrc_actuator, {static_cast<size_t>(self.model().nv())});
      })
      .def_prop_ro("qfrc_applied", [](MjoData& self) {
        return view(self.raw()->qfrc_applied, {static_cast<size_t>(self.model().nv())});
      })
      .def_prop_ro("xfrc_applied", [](MjoData& self) {
        return view(self.raw()->xfrc_applied, {static_cast<size_t>(self.model().nbody()), 6});
      })
      .def_prop_ro("xpos", [](MjoData& self) {
        return view(self.raw()->xpos, {static_cast<size_t>(self.model().nbody()), 3});
      })
      .def_prop_ro("xipos", [](MjoData& self) {
        return view(self.raw()->xipos, {static_cast<size_t>(self.model().nbody()), 3});
      })
      .def_prop_ro("xquat", [](MjoData& self) {
        return view(self.raw()->xquat, {static_cast<size_t>(self.model().nbody()), 4});
      })
      .def_prop_ro("xmat", [](MjoData& self) {
        return view(self.raw()->xmat, {static_cast<size_t>(self.model().nbody()), 9});
      })
      .def_prop_ro("ximat", [](MjoData& self) {
        return view(self.raw()->ximat, {static_cast<size_t>(self.model().nbody()), 9});
      })
      .def_prop_ro("cvel", [](MjoData& self) {
        return view(self.raw()->cvel, {static_cast<size_t>(self.model().nbody()), 6});
      })
      .def_prop_ro("sensordata", [](MjoData& self) {
        return view(self.raw()->sensordata, {static_cast<size_t>(self.model().nsensordata())});
      })
      .def_prop_ro("site_xmat", [](MjoData& self) {
        return view(self.raw()->site_xmat, {static_cast<size_t>(self.model().raw()->nsite), 9});
      })
      .def_prop_ro("wrench_buffer", [](MjoData& self) {
        return view(self.wrench_buffer(), {static_cast<size_t>(self.model().nbody()), 6});
      })
      .def_prop_ro("orbit", [](MjoData& self) { return OrbitView{&self}; })
      .def_prop_ro("frame", [](MjoData& self) { return FrameView{&self}; })
      .def_prop_ro("env", [](MjoData& self) { return EnvironmentView{&self}; })
      .def_prop_ro("actuators", [](MjoData& self) { return ActuatorView{&self}; })
      .def_prop_ro("sensors", [](MjoData& self) { return SensorDataView{&self}; });

  nb::class_<OrbitView>(m, "OrbitView")
      .def_prop_rw(
          "t",
          [](const OrbitView& self) { return self.data->orbit_instance()->t; },
          [](OrbitView& self, double value) { self.data->orbit_instance()->t = value; })
      .def_prop_rw(
          "R_eci",
          [](OrbitView& self) { return view(self.data->orbit_instance()->R_eci, {static_cast<size_t>(3)}); },
          [](OrbitView& self, const std::array<double, 3>& value) {
            copy_vec3(value, self.data->orbit_instance()->R_eci);
          })
      .def_prop_rw(
          "V_eci",
          [](OrbitView& self) { return view(self.data->orbit_instance()->V_eci, {static_cast<size_t>(3)}); },
          [](OrbitView& self, const std::array<double, 3>& value) {
            copy_vec3(value, self.data->orbit_instance()->V_eci);
          })
      .def_prop_ro("rk4_count", [](const OrbitView& self) {
        return self.data->orbit_instance()->orbit_rk4_count;
      });

  nb::class_<FrameView>(m, "FrameView")
      .def_prop_ro("C_LI", [](FrameView& self) {
        return view(self.data->orbit_instance()->C_LI, {static_cast<size_t>(3), 3});
      })
      .def_prop_ro("C_IL", [](FrameView& self) {
        return view(self.data->orbit_instance()->C_IL, {static_cast<size_t>(3), 3});
      })
      .def_prop_ro("omega_lvlh", [](FrameView& self) {
        return view(self.data->orbit_instance()->omega_lvlh, {static_cast<size_t>(3)});
      })
      .def_prop_ro("omega_dot_lvlh", [](FrameView& self) {
        return view(self.data->orbit_instance()->omega_dot_lvlh, {static_cast<size_t>(3)});
      });

  nb::class_<EnvironmentView>(m, "EnvironmentView")
      .def_prop_rw(
          "eclipse",
          [](const EnvironmentView& self) { return self.data->orbit_instance()->eclipse; },
          [](EnvironmentView& self, double value) { self.data->orbit_instance()->eclipse = value; })
      .def_prop_rw(
          "atm_density",
          [](const EnvironmentView& self) { return self.data->orbit_instance()->atm_density; },
          [](EnvironmentView& self, double value) { self.data->orbit_instance()->atm_density = value; })
      .def_prop_rw(
          "sun_vector_eci",
          [](EnvironmentView& self) {
            return view(self.data->orbit_instance()->sun_vector_eci, {static_cast<size_t>(3)});
          },
          [](EnvironmentView& self, const std::array<double, 3>& value) {
            copy_vec3(value, self.data->orbit_instance()->sun_vector_eci);
          })
      .def_prop_rw(
          "mag_field_eci",
          [](EnvironmentView& self) {
            return view(self.data->orbit_instance()->mag_field_eci, {static_cast<size_t>(3)});
          },
          [](EnvironmentView& self, const std::array<double, 3>& value) {
            copy_vec3(value, self.data->orbit_instance()->mag_field_eci);
          })
      .def_prop_ro("atmosphere_omega_eci", [](EnvironmentView& self) {
        return view(self.data->orbit_instance()->atmosphere_omega_eci, {static_cast<size_t>(3)});
      });

  nb::class_<ActuatorView>(m, "ActuatorView")
      .def_prop_ro("rw_speed", [](ActuatorView& self) {
        return view(self.data->rw_speed(), {self.data->model().reaction_wheels().size()});
      })
      .def_prop_ro("rw_momentum", [](ActuatorView& self) {
        return view(self.data->rw_momentum(), {self.data->model().reaction_wheels().size()});
      })
      .def_prop_ro("rw_torque_cmd", [](ActuatorView& self) {
        return view(self.data->rw_torque_cmd(), {self.data->model().reaction_wheels().size()});
      })
      .def_prop_ro("rw_inertia", [](ActuatorView& self) {
        std::vector<double> out;
        out.reserve(self.data->model().reaction_wheels().size());
        for (const auto& wheel : self.data->model().reaction_wheels()) {
          out.push_back(wheel.inertia);
        }
        return out;
      })
      .def_prop_ro("mtq_dipole_cmd", [](ActuatorView& self) {
        return view(self.data->mtq_dipole_cmd(), {self.data->model().magnetorquers().size()});
      })
      .def_prop_ro("thr_force_cmd", [](ActuatorView& self) {
        return view(self.data->thr_force_cmd(), {self.data->model().thrusters().size()});
      })
      .def_prop_ro("cmg_gimbal_angle", [](ActuatorView& self) {
        return view(self.data->cmg_gimbal_angle(), {self.data->model().cmgs().size()});
      })
      .def_prop_ro("cmg_gimbal_rate_cmd", [](ActuatorView& self) {
        return view(self.data->cmg_gimbal_rate_cmd(), {self.data->model().cmgs().size()});
      })
      .def_prop_ro("cmg_rotor_momentum", [](ActuatorView& self) {
        return view(self.data->cmg_rotor_momentum(), {self.data->model().cmgs().size()});
      })
      .def("update_rw_momentum", [](ActuatorView& self, nb::object) {
        const auto& wheels = self.data->model().reaction_wheels();
        for (std::size_t i = 0; i < wheels.size(); ++i) {
          self.data->rw_momentum()[i] = self.data->rw_speed()[i] * wheels[i].inertia;
        }
      }, "inertia"_a = nb::none());

  nb::class_<SensorDataView>(m, "SensorDataView")
      .def("measure", [](SensorDataView& self, const std::string& name, bool noisy, nb::object) {
        return self.data->measure_sensor(name, noisy);
      }, "name"_a, "noisy"_a = true, "rng"_a = nb::none())
      .def("bias", [](SensorDataView& self, const std::string& name) {
        auto it = self.data->sensor_biases().find(name);
        if (it == self.data->sensor_biases().end()) {
          return std::vector<double>{};
        }
        return it->second;
      })
      .def("descriptor", [](SensorDataView& self, const std::string& name) -> const SensorDescriptor& {
        return self.data->model().sensor(name);
      }, nb::rv_policy::reference_internal);

  m.def("mjo_forward", &mjo_forward, nb::call_guard<nb::gil_scoped_release>());
  m.def("mjo_step", &mjo_step, nb::call_guard<nb::gil_scoped_release>());
  m.def("mjo_state_size", &mjo_state_size);
  m.def("mjo_control_size", &mjo_control_size, "model"_a, "control_spec"_a);
  m.def("mjo_get_state", &get_state_vector);
  m.def("mjo_set_state", &set_state_array);
  m.def("mjo_rollout_native", &rollout_array);
}
