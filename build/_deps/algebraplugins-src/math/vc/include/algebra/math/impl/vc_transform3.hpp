/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/cmath.hpp"
#include "algebra/math/impl/vc_vector.hpp"
#include "algebra/qualifiers.hpp"

// Vc include(s).
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif  // MSVC
#include <Vc/Vc>
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // MSVC

// System include(s).
#include <cassert>

namespace algebra::vc::math {

using cmath::cross;

namespace internal {

/// 4x4 matrix type used by @c algebra::vc::math::transform3
template <typename scalar_t>
struct matrix44 {

  /// Equality operator between two matrices
  bool operator==(const matrix44 &rhs) const {
    return ((x == rhs.x).isFull() && (y == rhs.y).isFull() &&
            (z == rhs.z).isFull() && (t == rhs.t).isFull());
  }

  /// Data variables
  Vc::SimdArray<scalar_t, 4> x, y, z, t;

};  // struct matrix44

/// Functor used to access elements of Vc matrices
template <template <std::size_t> class array_t, typename scalar_t>
struct element_getter {

  /// Get const access to a matrix element
  ALGEBRA_HOST inline scalar_t operator()(const array_t<16> &m,
                                          unsigned int row,
                                          unsigned int col) const {

    // Make sure that the indices are valid.
    assert(row < 4);
    assert(col < 4);

    // Return the selected element.
    return m[col * 4 + row];
  }

  /// Get const access to a matrix element
  ALGEBRA_HOST inline scalar_t operator()(const matrix44<scalar_t> &m,
                                          unsigned int row,
                                          unsigned int col) const {

    // Make sure that the indices are valid.
    assert(row < 4);
    assert(col < 4);

    // Return the selected element.
    switch (row) {
      case 0:
        return m.x[col];
      case 1:
        return m.y[col];
      case 2:
        return m.z[col];
      case 3:
        return m.t[col];
      default:
        return 0;
    }
  }

};  // struct element_getter

}  // namespace internal

/** Transform wrapper class to ensure standard API within differnt plugins
 **/
template <template <typename, std::size_t> class array_t, typename scalar_t,
          typename vector3_t = array_t<scalar_t, 3>,
          typename point2_t = array_t<scalar_t, 2>>
struct transform3 {

  /// @name Type definitions for the struct
  /// @{

  /// Array type used by the transform
  template <std::size_t N>
  using array_type = array_t<scalar_t, N>;
  /// Scalar type used by the transform
  using scalar_type = scalar_t;

  /// 3-element "vector" type
  using vector3 = vector3_t;
  /// Point in 3D space
  using point3 = vector3;
  /// Point in 2D space
  using point2 = point2_t;

  /// 4x4 matrix type
  using matrix44 = internal::matrix44<scalar_type>;

  /// Function (object) used for accessing a matrix element
  using element_getter = internal::element_getter<array_type, scalar_type>;

  /// @}

  /// @name Data objects
  /// @{

  matrix44 _data;
  matrix44 _data_inv;

  /// @}

  /** Contructor with arguments: t, x, y, z
   *
   * @param t the translation (or origin of the new frame)
   * @param x the x axis of the new frame
   * @param y the y axis of the new frame
   * @param z the z axis of the new frame, normal vector for planes
   *
   **/
  ALGEBRA_HOST_DEVICE
  transform3(const vector3 &t, const vector3 &x, const vector3 &y,
             const vector3 &z, bool get_inverse = true) {
    _data.x = {x[0], x[1], x[2], 0.};
    _data.y = {y[0], y[1], y[2], 0.};
    _data.z = {z[0], z[1], z[2], 0.};
    _data.t = {t[0], t[1], t[2], 1.};

    if (get_inverse) {
      _data_inv = invert(_data);
    }
  }

  /** Contructor with arguments: t, z, x
   *
   * @param t the translation (or origin of the new frame)
   * @param z the z axis of the new frame, normal vector for planes
   * @param x the x axis of the new frame
   *
   * @note y will be constructed by cross product
   *
   **/
  ALGEBRA_HOST_DEVICE
  transform3(const vector3 &t, const vector3 &z, const vector3 &x,
             bool get_inverse = true)
      : transform3(t, x, cross(z, x), z, get_inverse) {}

  /** Constructor with arguments: translation
   *
   * @param t is the transform
   **/
  ALGEBRA_HOST_DEVICE
  transform3(const vector3 &t) {

    _data.x = {1., 0., 0., 0.};
    _data.y = {0., 1., 0., 0.};
    _data.z = {0., 0., 1., 0.};
    _data.t = {t[0], t[1], t[2], 1.};

    _data_inv = invert(_data);
  }

  /** Constructor with arguments: matrix
   *
   * @param m is the full 4x4 matrix
   **/
  ALGEBRA_HOST_DEVICE
  transform3(const matrix44 &m) {

    _data = m;
    _data_inv = invert(_data);
  }

  /** Constructor with arguments: matrix as std::aray of scalar
   *
   * @param ma is the full 4x4 matrix 16 array
   **/
  ALGEBRA_HOST_DEVICE
  transform3(const array_type<16> &ma) {

    _data.x = {ma[0], ma[4], ma[8], ma[12]};
    _data.y = {ma[1], ma[5], ma[9], ma[13]};
    _data.z = {ma[2], ma[6], ma[10], ma[14]};
    _data.t = {ma[3], ma[7], ma[11], ma[15]};

    _data_inv = invert(_data);
  }

  /** Constructor with arguments: identity
   *
   **/
  ALGEBRA_HOST_DEVICE
  transform3() {

    _data.x = {1., 0., 0., 0.};
    _data.y = {0., 1., 0., 0.};
    _data.z = {0., 0., 1., 0.};
    _data.t = {0., 0., 0., 1.};

    _data_inv = _data;
  }

  /** Default contructors */
  transform3(const transform3 &rhs) = default;
  ~transform3() = default;

  /** Equality operator */
  ALGEBRA_HOST_DEVICE
  inline bool operator==(const transform3 &rhs) const {
    return (_data == rhs._data);
  }

  /** The determinant of a 4x4 matrix
   *
   * @param m is the matrix
   *
   * @return a sacalar determinant - no checking done
   */
  ALGEBRA_HOST_DEVICE
  static inline scalar_type determinant(const matrix44 &m) {
    return m.t[0] * m.z[1] * m.y[2] * m.x[3] -
           m.z[0] * m.t[1] * m.y[2] * m.x[3] -
           m.t[0] * m.y[1] * m.z[2] * m.x[3] +
           m.y[0] * m.t[1] * m.z[2] * m.x[3] +
           m.z[0] * m.y[1] * m.t[2] * m.x[3] -
           m.y[0] * m.z[1] * m.t[2] * m.x[3] -
           m.t[0] * m.z[1] * m.x[2] * m.y[3] +
           m.z[0] * m.t[1] * m.x[2] * m.y[3] +
           m.t[0] * m.x[1] * m.z[2] * m.y[3] -
           m.x[0] * m.t[1] * m.z[2] * m.y[3] -
           m.z[0] * m.x[1] * m.t[2] * m.y[3] +
           m.x[0] * m.z[1] * m.t[2] * m.y[3] +
           m.t[0] * m.y[1] * m.x[2] * m.z[3] -
           m.y[0] * m.t[1] * m.x[2] * m.z[3] -
           m.t[0] * m.x[1] * m.y[2] * m.z[3] +
           m.x[0] * m.t[1] * m.y[2] * m.z[3] +
           m.y[0] * m.x[1] * m.t[2] * m.z[3] -
           m.x[0] * m.y[1] * m.t[2] * m.z[3] -
           m.z[0] * m.y[1] * m.x[2] * m.t[3] +
           m.y[0] * m.z[1] * m.x[2] * m.t[3] +
           m.z[0] * m.x[1] * m.y[2] * m.t[3] -
           m.x[0] * m.z[1] * m.y[2] * m.t[3] -
           m.y[0] * m.x[1] * m.z[2] * m.t[3] +
           m.x[0] * m.y[1] * m.z[2] * m.t[3];
  }

  /** The inverse of a 4x4 matrix
   *
   * @param m is the matrix
   *
   * @return an inverse matrix
   */
  ALGEBRA_HOST_DEVICE
  static inline matrix44 invert(const matrix44 &m) {
    matrix44 i;
    i.x[0] = m.z[1] * m.t[2] * m.y[3] - m.t[1] * m.z[2] * m.y[3] +
             m.t[1] * m.y[2] * m.z[3] - m.y[1] * m.t[2] * m.z[3] -
             m.z[1] * m.y[2] * m.t[3] + m.y[1] * m.z[2] * m.t[3];
    i.x[1] = m.t[1] * m.z[2] * m.x[3] - m.z[1] * m.t[2] * m.x[3] -
             m.t[1] * m.x[2] * m.z[3] + m.x[1] * m.t[2] * m.z[3] +
             m.z[1] * m.x[2] * m.t[3] - m.x[1] * m.z[2] * m.t[3];
    i.x[2] = m.y[1] * m.t[2] * m.x[3] - m.t[1] * m.y[2] * m.x[3] +
             m.t[1] * m.x[2] * m.y[3] - m.x[1] * m.t[2] * m.y[3] -
             m.y[1] * m.x[2] * m.t[3] + m.x[1] * m.y[2] * m.t[3];
    i.x[3] = m.z[1] * m.y[2] * m.x[3] - m.y[1] * m.z[2] * m.x[3] -
             m.z[1] * m.x[2] * m.y[3] + m.x[1] * m.z[2] * m.y[3] +
             m.y[1] * m.x[2] * m.z[3] - m.x[1] * m.y[2] * m.z[3];
    i.y[0] = m.t[0] * m.z[2] * m.y[3] - m.z[0] * m.t[2] * m.y[3] -
             m.t[0] * m.y[2] * m.z[3] + m.y[0] * m.t[2] * m.z[3] +
             m.z[0] * m.y[2] * m.t[3] - m.y[0] * m.z[2] * m.t[3];
    i.y[1] = m.z[0] * m.t[2] * m.x[3] - m.t[0] * m.z[2] * m.x[3] +
             m.t[0] * m.x[2] * m.z[3] - m.x[0] * m.t[2] * m.z[3] -
             m.z[0] * m.x[2] * m.t[3] + m.x[0] * m.z[2] * m.t[3];
    i.y[2] = m.t[0] * m.y[2] * m.x[3] - m.y[0] * m.t[2] * m.x[3] -
             m.t[0] * m.x[2] * m.y[3] + m.x[0] * m.t[2] * m.y[3] +
             m.y[0] * m.x[2] * m.t[3] - m.x[0] * m.y[2] * m.t[3];
    i.y[3] = m.y[0] * m.z[2] * m.x[3] - m.z[0] * m.y[2] * m.x[3] +
             m.z[0] * m.x[2] * m.y[3] - m.x[0] * m.z[2] * m.y[3] -
             m.y[0] * m.x[2] * m.z[3] + m.x[0] * m.y[2] * m.z[3];
    i.z[0] = m.z[0] * m.t[1] * m.y[3] - m.t[0] * m.z[1] * m.y[3] +
             m.t[0] * m.y[1] * m.z[3] - m.y[0] * m.t[1] * m.z[3] -
             m.z[0] * m.y[1] * m.t[3] + m.y[0] * m.z[1] * m.t[3];
    i.z[1] = m.t[0] * m.z[1] * m.x[3] - m.z[0] * m.t[1] * m.x[3] -
             m.t[0] * m.x[1] * m.z[3] + m.x[0] * m.t[1] * m.z[3] +
             m.z[0] * m.x[1] * m.t[3] - m.x[0] * m.z[1] * m.t[3];
    i.z[2] = m.y[0] * m.t[1] * m.x[3] - m.t[0] * m.y[1] * m.x[3] +
             m.t[0] * m.x[1] * m.y[3] - m.x[0] * m.t[1] * m.y[3] -
             m.y[0] * m.x[1] * m.t[3] + m.x[0] * m.y[1] * m.t[3];
    i.z[3] = m.z[0] * m.y[1] * m.x[3] - m.y[0] * m.z[1] * m.x[3] -
             m.z[0] * m.x[1] * m.y[3] + m.x[0] * m.z[1] * m.y[3] +
             m.y[0] * m.x[1] * m.z[3] - m.x[0] * m.y[1] * m.z[3];
    i.t[0] = m.t[0] * m.z[1] * m.y[2] - m.z[0] * m.t[1] * m.y[2] -
             m.t[0] * m.y[1] * m.z[2] + m.y[0] * m.t[1] * m.z[2] +
             m.z[0] * m.y[1] * m.t[2] - m.y[0] * m.z[1] * m.t[2];
    i.t[1] = m.z[0] * m.t[1] * m.x[2] - m.t[0] * m.z[1] * m.x[2] +
             m.t[0] * m.x[1] * m.z[2] - m.x[0] * m.t[1] * m.z[2] -
             m.z[0] * m.x[1] * m.t[2] + m.x[0] * m.z[1] * m.t[2];
    i.t[2] = m.t[0] * m.y[1] * m.x[2] - m.y[0] * m.t[1] * m.x[2] -
             m.t[0] * m.x[1] * m.y[2] + m.x[0] * m.t[1] * m.y[2] +
             m.y[0] * m.x[1] * m.t[2] - m.x[0] * m.y[1] * m.t[2];
    i.t[3] = m.y[0] * m.z[1] * m.x[2] - m.z[0] * m.y[1] * m.x[2] +
             m.z[0] * m.x[1] * m.y[2] - m.x[0] * m.z[1] * m.y[2] -
             m.y[0] * m.x[1] * m.z[2] + m.x[0] * m.y[1] * m.z[2];
    scalar_type idet = static_cast<scalar_type>(1.) / determinant(i);

    i.x *= idet;
    i.y *= idet;
    i.z *= idet;
    i.t *= idet;

    return i;
  }

  /** Rotate a vector into / from a frame
   *
   * @param m is the rotation matrix
   * @param v is the vector to be rotated
   */
  ALGEBRA_HOST_DEVICE
  static inline auto rotate(const matrix44 &m, const vector3 &v) {

    return m.x * v[0] + m.y * v[1] + m.z * v[2];
  }

  /** This method retrieves the rotation of a transform */
  ALGEBRA_HOST_DEVICE
  inline array_type<16> rotation() const {

    array_type<16> submatrix;
    for (unsigned int irow = 0; irow < 3; ++irow) {
      for (unsigned int icol = 0; icol < 3; ++icol) {
        submatrix[icol + irow * 4] = element_getter()(_data, irow, icol);
      }
    }
    return submatrix;
  }

  /** This method retrieves the translation of a transform */
  ALGEBRA_HOST_DEVICE
  inline point3 translation() const {
    return {_data.t[0], _data.t[1], _data.t[2]};
  }

  /** This method retrieves the 4x4 matrix of a transform */
  ALGEBRA_HOST_DEVICE
  inline const matrix44 &matrix() const { return _data; }

  /** This method retrieves the 4x4 matrix of an inverse transform */
  ALGEBRA_HOST_DEVICE
  inline const matrix44 &matrix_inverse() const { return _data_inv; }

  /** This method transform from a point from the local 3D cartesian frame
   *  to the global 3D cartesian frame
   *
   * @tparam point_type 3D point
   *
   * @param v is the point to be transformed
   *
   * @return a global point
   */
  template <typename point3_type>
  ALGEBRA_HOST_DEVICE inline point3_type point_to_global(
      const point3_type &v) const {

    auto g = _data.x * v[0] + _data.y * v[1] + _data.z * v[2] + _data.t;

    return point3_type{g[0], g[1], g[2]};
  }

  /** This method transform from a vector from the global 3D cartesian frame
   *  into the local 3D cartesian frame
   *
   * @tparam point_type 3D point
   *
   * @param v is the point to be transformed
   *
   * @return a local point
   */
  template <typename point3_type>
  ALGEBRA_HOST_DEVICE inline point3_type point_to_local(
      const point3_type &v) const {

    auto l = _data_inv.x * v[0] + _data_inv.y * v[1] + _data_inv.z * v[2] +
             _data_inv.t;

    return point3_type{l[0], l[1], l[2]};
  }

  /** This method transform from a vector from the local 3D cartesian frame
   *  to the global 3D cartesian frame
   *
   * @tparam vector_type 3D vector
   *
   * @param v is the vector to be transformed
   *
   * @return a vector in global coordinates
   */
  template <typename vector3_type>
  ALGEBRA_HOST_DEVICE inline vector3_type vector_to_global(
      const vector3_type &v) const {

    auto g = rotate(_data, v);
    return vector3_type{g[0], g[1], g[2]};
  }

  /** This method transform from a vector from the global 3D cartesian frame
   *  into the local 3D cartesian frame
   *
   * @tparam vector_type 3D vector
   *
   * @param v is the vector to be transformed
   *
   * @return a vector in global coordinates
   */
  template <typename vector3_type>
  ALGEBRA_HOST_DEVICE inline vector3_type vector_to_local(
      const vector3_type &v) const {

    auto l = rotate(_data_inv, v);

    return vector3_type{l[0], l[1], l[2]};
  }
};  // struct transform3

}  // namespace algebra::vc::math
