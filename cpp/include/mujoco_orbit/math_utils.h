#ifndef MUJOCO_ORBIT_MATH_UTILS_H_
#define MUJOCO_ORBIT_MATH_UTILS_H_

#include <algorithm>
#include <cmath>

namespace mujoco_orbit {
namespace detail {

inline constexpr double kPi = 3.141592653589793238462643383279502884;

inline double deg2rad(double degrees) { return degrees * (kPi / 180.0); }

inline double clamp(double value, double lower, double upper) {
  return std::max(lower, std::min(upper, value));
}

inline void zero3(double out[3]) {
  out[0] = 0.0;
  out[1] = 0.0;
  out[2] = 0.0;
}

inline void copy3(const double in[3], double out[3]) {
  out[0] = in[0];
  out[1] = in[1];
  out[2] = in[2];
}

inline double dot3(const double a[3], const double b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double norm3(const double v[3]) { return std::sqrt(dot3(v, v)); }

inline void scale3(const double in[3], double scale, double out[3]) {
  out[0] = scale * in[0];
  out[1] = scale * in[1];
  out[2] = scale * in[2];
}

inline void add3(const double a[3], const double b[3], double out[3]) {
  out[0] = a[0] + b[0];
  out[1] = a[1] + b[1];
  out[2] = a[2] + b[2];
}

inline void sub3(const double a[3], const double b[3], double out[3]) {
  out[0] = a[0] - b[0];
  out[1] = a[1] - b[1];
  out[2] = a[2] - b[2];
}

inline void add_scaled3(const double a[3], const double b[3], double scale, double out[3]) {
  out[0] = a[0] + scale * b[0];
  out[1] = a[1] + scale * b[1];
  out[2] = a[2] + scale * b[2];
}

inline void cross3(const double a[3], const double b[3], double out[3]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

inline void normalize3(const double in[3], double out[3]) {
  const double n = norm3(in);
  if (n == 0.0) {
    zero3(out);
    return;
  }
  scale3(in, 1.0 / n, out);
}

inline void set_identity3(double out[9]) {
  out[0] = 1.0;
  out[1] = 0.0;
  out[2] = 0.0;
  out[3] = 0.0;
  out[4] = 1.0;
  out[5] = 0.0;
  out[6] = 0.0;
  out[7] = 0.0;
  out[8] = 1.0;
}

inline void mat3_transpose(const double in[9], double out[9]) {
  out[0] = in[0];
  out[1] = in[3];
  out[2] = in[6];
  out[3] = in[1];
  out[4] = in[4];
  out[5] = in[7];
  out[6] = in[2];
  out[7] = in[5];
  out[8] = in[8];
}

inline void mat3_mul_vec(const double mat[9], const double vec[3], double out[3]) {
  out[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  out[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  out[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

inline void mat3_mul(const double left[9], const double right[9], double out[9]) {
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      out[3 * row + col] = left[3 * row + 0] * right[3 * 0 + col] +
                           left[3 * row + 1] * right[3 * 1 + col] +
                           left[3 * row + 2] * right[3 * 2 + col];
    }
  }
}

inline double mat3_det(const double mat[9]) {
  return mat[0] * (mat[4] * mat[8] - mat[5] * mat[7]) -
         mat[1] * (mat[3] * mat[8] - mat[5] * mat[6]) +
         mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);
}

}  // namespace detail
}  // namespace mujoco_orbit

#endif  // MUJOCO_ORBIT_MATH_UTILS_H_
