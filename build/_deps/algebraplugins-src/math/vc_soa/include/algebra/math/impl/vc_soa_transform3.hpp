/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/common.hpp"
#include "algebra/qualifiers.hpp"
#include "algebra/storage/impl/vc_soa_matrix44.hpp"
#include "algebra/storage/vector.hpp"

// Vc include(s).
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif  // MSVC
#include <Vc/Vc>
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // MSVC

// System include(s).
#include <array>
#include <cassert>
#include <type_traits>

namespace algebra::vc_soa::math {

using algebra::storage::operator*;
using algebra::storage::operator/;
using algebra::storage::operator-;
using algebra::storage::operator+;

/// Transform wrapper class to ensure standard API within differnt plugins
template <typename scalar_t>
struct transform3 {

  /// @name Type definitions for the struct
  /// @{

  /// Scalar type used by the transform
  using scalar_type = scalar_t;
  /// The type of the matrix elements (in this case: Vc::Vector)
  using value_type = Vc::Vector<scalar_t>;

  /// 3-element "vector" type (does not observe translations)
  using vector3 = storage::vector<3, value_type, std::array>;
  /// Point in 3D space (does observe translations)
  using point3 = vector3;
  /// Point in 2D space
  using point2 = storage::vector<2, value_type, std::array>;

  /// 4x4 matrix type
  using matrix44 = algebra::vc_soa::matrix44<scalar_type>;

  /// Function (object) used for accessing a matrix element
  using element_getter = algebra::vc_soa::element_getter<scalar_type>;

  /// @}

  /// @name Data objects
  /// @{

  matrix44 _data;
  matrix44 _data_inv;

  /// @}

  /// Default constructor: identity
  ALGEBRA_HOST_DEVICE
  transform3() = default;

  /// Contructor with arguments: t, x, y, z
  ///
  /// @param t the translation (or origin of the new frame)
  /// @param x the x axis of the new frame
  /// @param y the y axis of the new frame
  /// @param z the z axis of the new frame, normal vector for planes
  ALGEBRA_HOST_DEVICE
  transform3(const vector3 &t, const vector3 &x, const vector3 &y,
             const vector3 &z, [[maybe_unused]] bool get_inverse = true)
      : _data{x, y, z, t}, _data_inv{invert(_data)} {}

  /// Contructor with arguments: t, z, x
  ///
  /// @param t the translation (or origin of the new frame)
  /// @param z the z axis of the new frame, normal vector for planes
  /// @param x the x axis of the new frame
  ///
  /// @note y will be constructed by cross product
  ALGEBRA_HOST_DEVICE
  transform3(const vector3 &t, const vector3 &z, const vector3 &x,
             bool get_inverse = true)
      : transform3(t, x, cross(z, x), z, get_inverse) {}

  /// Constructor with arguments: translation
  ///
  /// @param t is the transform
  ALGEBRA_HOST_DEVICE
  transform3(const vector3 &t) : _data{t}, _data_inv{invert(_data)} {}

  /// Constructor with arguments: matrix
  ///
  /// @param m is the full 4x4 matrix with simd-vector elements
  ALGEBRA_HOST_DEVICE
  transform3(const matrix44 &m) : _data{m}, _data_inv{invert(_data)} {}

  /// Defaults
  transform3(const transform3 &rhs) = default;
  ~transform3() = default;

  /// Equality operator
  ALGEBRA_HOST_DEVICE
  inline constexpr bool operator==(const transform3 &rhs) const {
    return (_data == rhs._data);
  }

  /// Matrix access operator
  ALGEBRA_HOST_DEVICE
  inline const value_type &operator()(std::size_t i, std::size_t j) const {
    return element_getter{}(_data, i, j);
  }
  ALGEBRA_HOST_DEVICE
  inline value_type &operator()(std::size_t i, std::size_t j) {
    return element_getter{}(_data, i, j);
  }

  /// The determinant of a 4x4 matrix
  ///
  /// @param m is the matrix
  ///
  /// @return a sacalar determinant - no checking done
  ALGEBRA_HOST_DEVICE
  inline constexpr value_type determinant(const matrix44 &m) const {
    return -m.z[0] * m.y[1] * m.x[2] + m.y[0] * m.z[1] * m.x[2] +
           m.z[0] * m.x[1] * m.y[2] - m.x[0] * m.z[1] * m.y[2] -
           m.y[0] * m.x[1] * m.z[2] + m.x[0] * m.y[1] * m.z[2];
  }

  /// The inverse of a 4x4 matrix
  ///
  /// @param m is the matrix
  ///
  /// @return an inverse matrix
  ALGEBRA_HOST_DEVICE
  inline constexpr matrix44 invert(const matrix44 &m) const {
    matrix44 i;
    i.x[0] = -m.z[1] * m.y[2] + m.y[1] * m.z[2];
    i.x[1] = m.z[1] * m.x[2] - m.x[1] * m.z[2];
    i.x[2] = -m.y[1] * m.x[2] + m.x[1] * m.y[2];
    // i.x[3] = 0;
    i.y[0] = m.z[0] * m.y[2] - m.y[0] * m.z[2];
    i.y[1] = -m.z[0] * m.x[2] + m.x[0] * m.z[2];
    i.y[2] = m.y[0] * m.x[2] - m.x[0] * m.y[2];
    // i.y[3] = 0;
    i.z[0] = -m.z[0] * m.y[1] + m.y[0] * m.z[1];
    i.z[1] = m.z[0] * m.x[1] - m.x[0] * m.z[1];
    i.z[2] = -m.y[0] * m.x[1] + m.x[0] * m.y[1];
    // i.z[3] = 0;
    i.t[0] = m.t[0] * m.z[1] * m.y[2] - m.z[0] * m.t[1] * m.y[2] -
             m.t[0] * m.y[1] * m.z[2] + m.y[0] * m.t[1] * m.z[2] +
             m.z[0] * m.y[1] * m.t[2] - m.y[0] * m.z[1] * m.t[2];
    i.t[1] = m.z[0] * m.t[1] * m.x[2] - m.t[0] * m.z[1] * m.x[2] +
             m.t[0] * m.x[1] * m.z[2] - m.x[0] * m.t[1] * m.z[2] -
             m.z[0] * m.x[1] * m.t[2] + m.x[0] * m.z[1] * m.t[2];
    i.t[2] = m.t[0] * m.y[1] * m.x[2] - m.y[0] * m.t[1] * m.x[2] -
             m.t[0] * m.x[1] * m.y[2] + m.x[0] * m.t[1] * m.y[2] +
             m.y[0] * m.x[1] * m.t[2] - m.x[0] * m.y[1] * m.t[2];
    // i.t[3] = 1;
    const value_type idet{value_type::One() / determinant(i)};

    i.x = i.x * idet;
    i.y = i.y * idet;
    i.z = i.z * idet;
    i.t = i.t * idet;

    return i;
  }

  /// Rotate a vector into / from a frame
  ///
  /// @param m is the rotation matrix
  /// @param v is the vector to be rotated
  ALGEBRA_HOST_DEVICE
  inline constexpr auto rotate(const matrix44 &m, const vector3 &v) const {

    return m.x * v[0] + m.y * v[1] + m.z * v[2];
  }

  /// @returns the translation of the transform
  ALGEBRA_HOST_DEVICE
  inline point3 translation() const { return _data.t; }

  /// @returns the 4x4 matrix of the transform
  ALGEBRA_HOST_DEVICE
  inline const matrix44 &matrix() const { return _data; }

  /// @returns the 4x4 matrix of the inverse transform
  ALGEBRA_HOST_DEVICE
  inline const matrix44 &matrix_inverse() const { return _data_inv; }

  /// This method transforms a point from the local 3D cartesian frame
  /// to the global 3D cartesian frame
  ///
  /// @tparam point3_t 3D point
  ///
  /// @param v is the point to be transformed
  ///
  /// @return a global point
  template <
      typename point3_t,
      std::enable_if_t<std::is_convertible_v<point3_t, point3>, bool> = true>
  ALGEBRA_HOST_DEVICE inline point3 point_to_global(const point3_t &v) const {

    return rotate(_data, v) + _data.t;
  }

  /// This method transforms a point from the global 3D cartesian frame
  /// into the local 3D cartesian frame
  ///
  /// @tparam point3_t 3D point
  ///
  /// @param v is the point to be transformed
  ///
  /// @return a local point
  template <
      typename point3_t,
      std::enable_if_t<std::is_convertible_v<point3_t, point3>, bool> = true>
  ALGEBRA_HOST_DEVICE inline point3 point_to_local(const point3_t &v) const {

    return rotate(_data_inv, v) + _data_inv.t;
  }

  /// This method transforms a vector from the local 3D cartesian frame
  /// to the global 3D cartesian frame
  ///
  /// @tparam vector3_t 3D vector
  ///
  /// @param v is the vector to be transformed
  ///
  /// @return a vector in global coordinates
  template <
      typename vector3_t,
      std::enable_if_t<std::is_convertible_v<vector3_t, vector3>, bool> = true>
  ALGEBRA_HOST_DEVICE inline vector3 vector_to_global(
      const vector3_t &v) const {

    return rotate(_data, v);
  }

  /// This method transforms a vector from the global 3D cartesian frame
  /// into the local 3D cartesian frame
  ///
  /// @tparam vector3_t 3D vector
  ///
  /// @param v is the vector to be transformed
  ///
  /// @return a vector in local coordinates
  template <
      typename vector3_t,
      std::enable_if_t<std::is_convertible_v<vector3_t, vector3>, bool> = true>
  ALGEBRA_HOST_DEVICE inline vector3 vector_to_local(const vector3_t &v) const {

    return rotate(_data_inv, v);
  }
};  // struct transform3

}  // namespace algebra::vc_soa::math
