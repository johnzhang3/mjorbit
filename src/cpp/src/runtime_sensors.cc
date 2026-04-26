#include "mujoco_orbit/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "mujoco_orbit/math_utils.h"

namespace mujoco_orbit {
namespace {

void normalized3(const double in[3], double out[3], const std::string& label) {
  const double norm = detail::norm3(in);
  if (norm < 1.0e-12) {
    throw std::runtime_error(label + " must be non-zero");
  }
  detail::scale3(in, 1.0 / norm, out);
}

void rotvec_to_matrix(const double rotvec[3], double out[9]) {
  const double theta = detail::norm3(rotvec);
  if (theta < 1.0e-12) {
    out[0] = 1.0;
    out[1] = -rotvec[2];
    out[2] = rotvec[1];
    out[3] = rotvec[2];
    out[4] = 1.0;
    out[5] = -rotvec[0];
    out[6] = -rotvec[1];
    out[7] = rotvec[0];
    out[8] = 1.0;
    return;
  }

  const double axis[3] = {rotvec[0] / theta, rotvec[1] / theta, rotvec[2] / theta};
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  const double one_minus_c = 1.0 - c;
  out[0] = c + axis[0] * axis[0] * one_minus_c;
  out[1] = axis[0] * axis[1] * one_minus_c - axis[2] * s;
  out[2] = axis[0] * axis[2] * one_minus_c + axis[1] * s;
  out[3] = axis[1] * axis[0] * one_minus_c + axis[2] * s;
  out[4] = c + axis[1] * axis[1] * one_minus_c;
  out[5] = axis[1] * axis[2] * one_minus_c - axis[0] * s;
  out[6] = axis[2] * axis[0] * one_minus_c - axis[1] * s;
  out[7] = axis[2] * axis[1] * one_minus_c + axis[0] * s;
  out[8] = c + axis[2] * axis[2] * one_minus_c;
}

void rotate_axis(std::vector<double>* measurement, const double rotvec[3]) {
  if (measurement->size() != 3) {
    return;
  }
  double matrix[9];
  rotvec_to_matrix(rotvec, matrix);
  const double axis[3] = {(*measurement)[0], (*measurement)[1], (*measurement)[2]};
  double rotated[3];
  detail::mat3_mul_vec(matrix, axis, rotated);
  normalized3(rotated, rotated, "Noisy axis measurement");
  *measurement = {rotated[0], rotated[1], rotated[2]};
}

void rotvec_to_quat(const double rotvec[3], double out[4]) {
  const double theta = detail::norm3(rotvec);
  if (theta < 1.0e-12) {
    out[0] = 1.0;
    out[1] = 0.5 * rotvec[0];
    out[2] = 0.5 * rotvec[1];
    out[3] = 0.5 * rotvec[2];
  } else {
    const double half = 0.5 * theta;
    const double scale = std::sin(half) / theta;
    out[0] = std::cos(half);
    out[1] = scale * rotvec[0];
    out[2] = scale * rotvec[1];
    out[3] = scale * rotvec[2];
  }
  const double norm = std::sqrt(
      out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]);
  for (int i = 0; i < 4; ++i) {
    out[i] /= norm;
  }
}

void quat_mul(const double left[4], const double right[4], double out[4]) {
  out[0] = left[0] * right[0] - left[1] * right[1] -
           left[2] * right[2] - left[3] * right[3];
  out[1] = left[0] * right[1] + left[1] * right[0] +
           left[2] * right[3] - left[3] * right[2];
  out[2] = left[0] * right[2] - left[1] * right[3] +
           left[2] * right[0] + left[3] * right[1];
  out[3] = left[0] * right[3] + left[1] * right[2] -
           left[2] * right[1] + left[3] * right[0];
}

void rotate_quaternion(std::vector<double>* measurement, const double rotvec[3]) {
  if (measurement->size() != 4) {
    return;
  }
  double delta[4];
  rotvec_to_quat(rotvec, delta);
  const double quat[4] = {(*measurement)[0], (*measurement)[1], (*measurement)[2],
                          (*measurement)[3]};
  double rotated[4];
  quat_mul(delta, quat, rotated);
  const double norm = std::sqrt(rotated[0] * rotated[0] + rotated[1] * rotated[1] +
                                rotated[2] * rotated[2] + rotated[3] * rotated[3]);
  *measurement = {rotated[0] / norm, rotated[1] / norm, rotated[2] / norm,
                  rotated[3] / norm};
}

void apply_rotvec_sensor_effect(
    const SensorDescriptor& descriptor,
    std::vector<double>* measurement,
    const double rotvec[3]) {
  if (descriptor.datatype == mjDATATYPE_AXIS) {
    rotate_axis(measurement, rotvec);
  } else if (descriptor.datatype == mjDATATYPE_QUATERNION) {
    rotate_quaternion(measurement, rotvec);
  }
}

double random_uniform01(std::uint64_t* state) {
  *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<double>((*state >> 11) & ((1ULL << 53) - 1)) /
         static_cast<double>(1ULL << 53);
}

double random_normal(std::uint64_t* state, double sigma) {
  if (sigma <= 0.0) return 0.0;
  const double u1 = std::max(random_uniform01(state), 1.0e-12);
  const double u2 = random_uniform01(state);
  return sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * detail::kPi * u2);
}

}  // namespace

void MjoData::initialize_sensor_biases(std::optional<std::uint64_t> rng_seed) {
  rng_state_ = rng_seed.value_or(0x9e3779b97f4a7c15ULL);
  for (const SensorDescriptor& descriptor : model_->sensors().descriptors) {
    const std::vector<double>* sigma = nullptr;
    if (!descriptor.additive_bias_sigma.empty()) {
      sigma = &descriptor.additive_bias_sigma;
    } else if (!descriptor.angular_bias_sigma.empty()) {
      sigma = &descriptor.angular_bias_sigma;
    }
    if (!sigma) {
      continue;
    }
    std::vector<double> bias(sigma->size(), 0.0);
    for (std::size_t i = 0; i < sigma->size(); ++i) {
      bias[i] = random_normal(&rng_state_, (*sigma)[i]);
    }
    sensor_biases_[descriptor.name] = std::move(bias);
  }
}

std::vector<double> MjoData::measure_sensor(const std::string& name, bool noisy) {
  const SensorDescriptor& descriptor = model_->sensor(name);
  std::vector<double> measurement(static_cast<std::size_t>(descriptor.dim), 0.0);
  for (int i = 0; i < descriptor.dim; ++i) {
    measurement[static_cast<std::size_t>(i)] = data_->sensordata[descriptor.adr + i];
  }
  if (!noisy) {
    return measurement;
  }
  auto bias_it = sensor_biases_.find(name);
  if (bias_it != sensor_biases_.end()) {
    const std::vector<double>& bias = bias_it->second;
    if ((descriptor.datatype == mjDATATYPE_AXIS || descriptor.datatype == mjDATATYPE_QUATERNION) &&
        bias.size() >= 3) {
      const double rotvec[3] = {bias[0], bias[1], bias[2]};
      apply_rotvec_sensor_effect(descriptor, &measurement, rotvec);
    } else {
      for (std::size_t i = 0; i < measurement.size() && i < bias.size(); ++i) {
        measurement[i] += bias[i];
      }
    }
  }
  if (descriptor.noise > 0.0) {
    if (descriptor.datatype == mjDATATYPE_AXIS || descriptor.datatype == mjDATATYPE_QUATERNION) {
      const double rotvec[3] = {
          random_normal(&rng_state_, descriptor.noise),
          random_normal(&rng_state_, descriptor.noise),
          random_normal(&rng_state_, descriptor.noise),
      };
      apply_rotvec_sensor_effect(descriptor, &measurement, rotvec);
    } else {
      for (double& value : measurement) {
        value += random_normal(&rng_state_, descriptor.noise);
      }
      if (descriptor.datatype == mjDATATYPE_POSITIVE) {
        for (double& value : measurement) {
          value = std::max(0.0, value);
        }
      }
    }
  }
  if (descriptor.cutoff > 0.0 &&
      descriptor.datatype != mjDATATYPE_AXIS &&
      descriptor.datatype != mjDATATYPE_QUATERNION) {
    for (double& value : measurement) {
      const double lower = descriptor.datatype == mjDATATYPE_POSITIVE ? 0.0 : -descriptor.cutoff;
      value = detail::clamp(value, lower, descriptor.cutoff);
    }
  }
  return measurement;
}

}  // namespace mujoco_orbit
