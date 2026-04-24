#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <mujoco/mujoco.h>

#include "mujoco_orbit/constants.h"
#include "mujoco_orbit/environment.h"
#include "mujoco_orbit/lvlh.h"
#include "mujoco_orbit/orbit_state.h"
#include "mujoco_orbit/sensors_plugin.h"
#include "orbit_instance.h"

namespace {

constexpr int kWorkers = 8;
constexpr int kSteps = 1000;

const char kThreadedXml[] = R"xml(
<mujoco model="threaded_plugin">
  <size nuser_sensor="4"/>
  <option timestep="0.005" gravity="0 0 0"/>
  <extension>
    <plugin plugin="mujoco_orbit.orbit"/>
  </extension>
  <worldbody>
    <body name="spacecraft" pos="0 0 0">
      <freejoint/>
      <geom type="box" size="0.5 0.5 0.5" mass="100"/>
      <site name="sun_head" pos="0 0 0.2" quat="1 0 0 0"/>
      <plugin plugin="mujoco_orbit.orbit">
        <config key="use_j2" value="false"/>
      </plugin>
    </body>
  </worldbody>
  <sensor>
    <user
      name="orbit_sun_body"
      objtype="site"
      objname="sun_head"
      datatype="axis"
      needstage="pos"
      dim="3"
    />
  </sensor>
</mujoco>
)xml";

struct MjModelDeleter {
  void operator()(mjModel* model) const { mj_deleteModel(model); }
};

struct MjDataDeleter {
  void operator()(mjData* data) const { mj_deleteData(data); }
};

using ModelPtr = std::unique_ptr<mjModel, MjModelDeleter>;
using DataPtr = std::unique_ptr<mjData, MjDataDeleter>;

struct Result {
  std::vector<double> qpos;
  std::vector<double> qvel;
  std::vector<double> sensordata;
  std::array<double, 3> R_eci{};
  std::array<double, 3> V_eci{};
  double t = 0.0;
};

struct WorkerData {
  DataPtr data;
  std::array<mujoco_orbit::OrbitSensorDescriptorNative, 1> sensors{};
  Result result;
};

std::string write_xml_file() {
  const char* tmpdir_env = std::getenv("TMPDIR");
  const std::string tmpdir = tmpdir_env && tmpdir_env[0] != '\0' ? tmpdir_env : "/tmp";
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::string path =
      tmpdir + "/mujoco_orbit_threaded_plugin_" + std::to_string(stamp) + ".xml";
  std::ofstream file(path);
  file << kThreadedXml;
  return path;
}

ModelPtr load_model(const std::string& path) {
  char error[1024] = {0};
  mjModel* model = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
  if (!model) {
    std::cerr << "mj_loadXML failed: " << error << "\n";
    return nullptr;
  }
  return ModelPtr(model);
}

mujoco_orbit::OrbitInstance* orbit_instance(const mjModel* model, mjData* data) {
  const int body_id = mj_name2id(model, mjOBJ_BODY, "spacecraft");
  if (body_id < 0) {
    return nullptr;
  }
  const int instance = model->body_plugin[body_id];
  if (instance < 0) {
    return nullptr;
  }
  return reinterpret_cast<mujoco_orbit::OrbitInstance*>(data->plugin_data[instance]);
}

void refresh_caches(mujoco_orbit::OrbitInstance* inst) {
  mujoco_orbit::OrbitState orbit{};
  std::memcpy(orbit.R_eci, inst->R_eci, sizeof(orbit.R_eci));
  std::memcpy(orbit.V_eci, inst->V_eci, sizeof(orbit.V_eci));
  orbit.t = inst->t;

  mujoco_orbit::FrameCache frame{};
  mujoco_orbit::update_frame_cache(orbit, &frame, inst->use_j2 != 0);
  std::memcpy(inst->C_LI, frame.C_LI, sizeof(inst->C_LI));
  std::memcpy(inst->C_IL, frame.C_IL, sizeof(inst->C_IL));
  std::memcpy(inst->omega_lvlh, frame.omega_lvlh, sizeof(inst->omega_lvlh));
  std::memcpy(inst->omega_dot_lvlh, frame.omega_dot_lvlh, sizeof(inst->omega_dot_lvlh));

  mujoco_orbit::EnvironmentCache env{};
  mujoco_orbit::update_environment_cache(orbit, frame, &env);
  std::memcpy(inst->sun_vector_eci, env.sun_vector_eci, sizeof(inst->sun_vector_eci));
  std::memcpy(inst->mag_field_eci, env.mag_field_eci, sizeof(inst->mag_field_eci));
  std::memcpy(
      inst->atmosphere_omega_eci,
      env.atmosphere_omega_eci,
      sizeof(inst->atmosphere_omega_eci));
  inst->atm_density = env.atm_density;
  inst->eclipse = env.eclipse;
}

bool init_worker(const mjModel* model, WorkerData* worker) {
  worker->data.reset(mj_makeData(model));
  if (!worker->data) {
    std::cerr << "mj_makeData failed\n";
    return false;
  }

  auto* inst = orbit_instance(model, worker->data.get());
  if (!inst) {
    std::cerr << "orbit plugin instance not found\n";
    return false;
  }

  const double r = mujoco_orbit::kREarth + 400.0;
  inst->R_eci[0] = r;
  inst->R_eci[1] = 0.0;
  inst->R_eci[2] = 0.0;
  inst->V_eci[0] = 0.0;
  inst->V_eci[1] = std::sqrt(mujoco_orbit::kGmEarth / r);
  inst->V_eci[2] = 0.0;
  inst->t = 0.0;
  inst->use_j2 = 0;
  refresh_caches(inst);

  const int sensor_id = mj_name2id(model, mjOBJ_SENSOR, "orbit_sun_body");
  const int site_id = mj_name2id(model, mjOBJ_SITE, "sun_head");
  if (sensor_id < 0 || site_id < 0) {
    std::cerr << "test sensor/site not found\n";
    return false;
  }
  worker->sensors[0].sensor_id = sensor_id;
  worker->sensors[0].kind = mujoco_orbit::kOrbitSensorSun;
  worker->sensors[0].site_id = site_id;
  worker->sensors[0].adr = model->sensor_adr[sensor_id];
  worker->sensors[0].dim = model->sensor_dim[sensor_id];
  worker->sensors[0].reference_eci[0] = 0.0;
  worker->sensors[0].reference_eci[1] = 0.0;
  worker->sensors[0].reference_eci[2] = 0.0;
  inst->num_orbit_sensors = static_cast<int>(worker->sensors.size());
  inst->orbit_sensors = worker->sensors.data();

  mjData* data = worker->data.get();
  data->qpos[0] = 10.0;
  data->qpos[1] = -3.0;
  data->qpos[2] = 2.0;
  data->qvel[0] = 0.01;
  data->qvel[1] = -0.02;
  data->qvel[2] = 0.005;
  data->qvel[3] = 0.02;
  data->qvel[4] = -0.01;
  data->qvel[5] = 0.03;
  mj_forward(model, data);
  return true;
}

Result capture_result(const mjModel* model, mjData* data) {
  Result result;
  result.qpos.assign(data->qpos, data->qpos + model->nq);
  result.qvel.assign(data->qvel, data->qvel + model->nv);
  result.sensordata.assign(data->sensordata, data->sensordata + model->nsensordata);

  auto* inst = orbit_instance(model, data);
  if (inst) {
    std::copy(inst->R_eci, inst->R_eci + 3, result.R_eci.begin());
    std::copy(inst->V_eci, inst->V_eci + 3, result.V_eci.begin());
    result.t = inst->t;
  }
  return result;
}

void run_steps(const mjModel* model, WorkerData* worker) {
  for (int i = 0; i < kSteps; ++i) {
    mj_step(model, worker->data.get());
  }
  worker->result = capture_result(model, worker->data.get());
}

bool vector_equal(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

bool result_equal(const Result& a, const Result& b) {
  return vector_equal(a.qpos, b.qpos) &&
         vector_equal(a.qvel, b.qvel) &&
         vector_equal(a.sensordata, b.sensordata) &&
         a.R_eci == b.R_eci &&
         a.V_eci == b.V_eci &&
         a.t == b.t;
}

bool plugin_instances_are_distinct(const mjModel* model, const std::vector<WorkerData>& workers) {
  std::vector<const mujoco_orbit::OrbitInstance*> instances;
  instances.reserve(workers.size());
  for (const auto& worker : workers) {
    const auto* inst = orbit_instance(model, worker.data.get());
    if (!inst) {
      return false;
    }
    instances.push_back(inst);
  }

  for (std::size_t i = 0; i < instances.size(); ++i) {
    for (std::size_t j = i + 1; j < instances.size(); ++j) {
      if (instances[i] == instances[j]) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  const std::string xml_path = write_xml_file();
  ModelPtr model = load_model(xml_path);
  std::remove(xml_path.c_str());
  if (!model) {
    return 1;
  }

  WorkerData expected;
  if (!init_worker(model.get(), &expected)) {
    return 1;
  }
  run_steps(model.get(), &expected);

  std::vector<WorkerData> workers(kWorkers);
  for (auto& worker : workers) {
    if (!init_worker(model.get(), &worker)) {
      return 1;
    }
  }
  if (!plugin_instances_are_distinct(model.get(), workers)) {
    std::cerr << "workers unexpectedly share orbit plugin state\n";
    return 1;
  }

  std::vector<std::thread> threads;
  threads.reserve(workers.size());
  for (auto& worker : workers) {
    threads.emplace_back(run_steps, model.get(), &worker);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  for (const auto& worker : workers) {
    if (!result_equal(worker.result, expected.result)) {
      std::cerr << "threaded rollout diverged from sequential baseline\n";
      return 1;
    }
  }

  std::cout << "threaded plugin rollouts matched sequential baseline\n";
  return 0;
}
