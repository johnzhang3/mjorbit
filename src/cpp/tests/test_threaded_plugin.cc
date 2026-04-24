#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <mujoco/mujoco.h>

#include "mujoco_orbit/constants.h"
#include "mujoco_orbit/orbit_cache.h"
#include "mujoco_orbit/sensors_plugin.h"
#include "orbit_instance.h"

extern "C" int mjo_rollout(
    const mjModel* m,
    mjData* d,
    int orbit_plugin_instance,
    int nbatch,
    int nstep,
    unsigned int control_spec,
    int mjo_state_size,
    int mjo_control_size,
    const mjtNum* initial_state,
    const mjtNum* initial_warmstart,
    const mjtNum* control,
    mjtNum* state,
    mjtNum* sensordata);

namespace {

constexpr int kWorkers = 8;
constexpr int kSteps = 1000;
constexpr int kRolloutSteps = 128;

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
  int rollout_status = 0;
  std::vector<double> rollout_state;
  std::vector<double> rollout_sensordata;
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

int orbit_plugin_instance(const mjModel* model) {
  const int body_id = mj_name2id(model, mjOBJ_BODY, "spacecraft");
  if (body_id < 0) {
    return -1;
  }
  return model->body_plugin[body_id];
}

mujoco_orbit::OrbitInstance* orbit_instance(const mjModel* model, mjData* data) {
  const int instance = orbit_plugin_instance(model);
  if (instance < 0) {
    return nullptr;
  }
  return reinterpret_cast<mujoco_orbit::OrbitInstance*>(data->plugin_data[instance]);
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
  mujoco_orbit::refresh_orbit_caches(inst);

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

int mjo_state_size(const mjModel* model) {
  return mj_stateSize(model, mjSTATE_FULLPHYSICS) + 7;
}

int mjo_control_size(const mjModel* model) {
  return mj_stateSize(model, mjSTATE_CTRL);
}

std::vector<double> pack_mjo_state(const mjModel* model, mjData* data) {
  const int full_state_size = mj_stateSize(model, mjSTATE_FULLPHYSICS);
  std::vector<double> state(mjo_state_size(model), 0.0);
  mj_getState(model, data, state.data(), mjSTATE_FULLPHYSICS);

  auto* inst = orbit_instance(model, data);
  if (!inst) {
    return state;
  }
  double* tail = state.data() + full_state_size;
  std::copy(inst->R_eci, inst->R_eci + 3, tail);
  tail += 3;
  std::copy(inst->V_eci, inst->V_eci + 3, tail);
  tail += 3;
  *tail = inst->t;
  return state;
}

void run_native_rollout(const mjModel* model, WorkerData* worker) {
  std::vector<double> initial_state = pack_mjo_state(model, worker->data.get());
  worker->rollout_state.assign(kRolloutSteps * mjo_state_size(model), 0.0);
  worker->rollout_sensordata.assign(kRolloutSteps * model->nsensordata, 0.0);
  worker->rollout_status = mjo_rollout(
      model,
      worker->data.get(),
      orbit_plugin_instance(model),
      1,
      kRolloutSteps,
      mjSTATE_CTRL,
      mjo_state_size(model),
      mjo_control_size(model),
      initial_state.data(),
      nullptr,
      nullptr,
      worker->rollout_state.data(),
      worker->rollout_sensordata.data());
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

bool rollout_equal(const WorkerData& a, const WorkerData& b) {
  return a.rollout_status == b.rollout_status &&
         vector_equal(a.rollout_state, b.rollout_state) &&
         vector_equal(a.rollout_sensordata, b.rollout_sensordata) &&
         result_equal(a.result, b.result);
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

int run_threaded_step_test(const mjModel* model) {
  WorkerData expected;
  if (!init_worker(model, &expected)) {
    return 1;
  }
  run_steps(model, &expected);

  std::vector<WorkerData> workers(kWorkers);
  for (auto& worker : workers) {
    if (!init_worker(model, &worker)) {
      return 1;
    }
  }
  if (!plugin_instances_are_distinct(model, workers)) {
    std::cerr << "workers unexpectedly share orbit plugin state\n";
    return 1;
  }

  std::vector<std::thread> threads;
  threads.reserve(workers.size());
  for (auto& worker : workers) {
    threads.emplace_back(run_steps, model, &worker);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  for (const auto& worker : workers) {
    if (!result_equal(worker.result, expected.result)) {
      std::cerr << "threaded mj_step rollout diverged from sequential baseline\n";
      return 1;
    }
  }

  std::cout << "threaded mj_step rollouts matched sequential baseline\n";
  return 0;
}

int run_threaded_rollout_test(const mjModel* model) {
  WorkerData expected;
  if (!init_worker(model, &expected)) {
    return 1;
  }
  run_native_rollout(model, &expected);
  if (expected.rollout_status != 0) {
    std::cerr << "sequential mjo_rollout failed with status " << expected.rollout_status
              << "\n";
    return 1;
  }

  std::vector<WorkerData> workers(kWorkers);
  for (auto& worker : workers) {
    if (!init_worker(model, &worker)) {
      return 1;
    }
  }
  if (!plugin_instances_are_distinct(model, workers)) {
    std::cerr << "rollout workers unexpectedly share orbit plugin state\n";
    return 1;
  }

  std::vector<std::thread> threads;
  threads.reserve(workers.size());
  for (auto& worker : workers) {
    threads.emplace_back(run_native_rollout, model, &worker);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  for (const auto& worker : workers) {
    if (worker.rollout_status != 0) {
      std::cerr << "threaded mjo_rollout failed with status " << worker.rollout_status
                << "\n";
      return 1;
    }
    if (!rollout_equal(worker, expected)) {
      std::cerr << "threaded mjo_rollout diverged from sequential baseline\n";
      return 1;
    }
  }

  std::cout << "threaded mjo_rollout calls matched sequential baseline\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string xml_path = write_xml_file();
  ModelPtr model = load_model(xml_path);
  std::remove(xml_path.c_str());
  if (!model) {
    return 1;
  }

  if (argc > 1) {
    const std::string mode = argv[1];
    if (mode == "--step-only") {
      return run_threaded_step_test(model.get());
    }
    if (mode == "--rollout-only") {
      return run_threaded_rollout_test(model.get());
    }
    std::cerr << "unknown mode: " << mode << "\n";
    return 1;
  }

  if (run_threaded_step_test(model.get()) != 0) {
    return 1;
  }
  return run_threaded_rollout_test(model.get());
}
