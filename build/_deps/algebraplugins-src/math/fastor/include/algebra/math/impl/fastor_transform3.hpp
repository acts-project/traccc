/** Algebra plugins, part of the ACTS project
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/qualifiers.hpp"

// Fastor include(s).
#ifdef _MSC_VER
#pragma warning(disable : 4244 4701 4702)
#endif  // MSVC
#include <Fastor/Fastor.h>
#ifdef _MSC_VER
#pragma warning(default : 4244 4701 4702)
#endif  // MSVC

// System include(s).
#include <cstddef>

namespace algebra::fastor::math {

/** Transform wrapper class to ensure standard API within different plugins
 *
 **/
template <typename scalar_t, typename matrix_actor_t>
struct transform3 {
  /// @name Type definitions for the struct
  /// @{

  /// Array type used by the transform
  template <std::size_t N>
  using array_type = Fastor::Tensor<scalar_t, N>;
  /// Scalar type used by the transform
  using scalar_type = scalar_t;

  /// 3-element "vector" type
  using vector3 = array_type<3>;
  /// Point in 3D space
  using point3 = vector3;
  /// Point in 2D space
  using point2 = array_type<2>;

  /// 4x4 matrix type
  using matrix44 = Fastor::Tensor<scalar_type, 4UL, 4UL>;

  /// Function (object) used for accessing a matrix element
  using element_getter = algebra::fastor::math::element_getter<scalar_t>;

  /// Size type
  using size_type = typename matrix_actor_t::size_ty;

  /// Matrix actor
  using matrix_actor = matrix_actor_t;

  /// 2D Matrix type
  template <size_type ROWS, size_type COLS>
  using matrix_type = typename matrix_actor::template matrix_type<ROWS, COLS>;

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

    // The matrix needs to be initialized to the identity matrix first. We only
    // modify the top 4x3 portion of the matrix, so it doesn't matter what
    // values it initially had. However, the bottom row is required to have the
    // values of [0, 0, 0, 1], so that's why we set `_data` to the identity
    // matrix first.
    _data.eye2();

    _data(Fastor::fseq<0, 3>(), 0) = x;
    _data(Fastor::fseq<0, 3>(), 1) = y;
    _data(Fastor::fseq<0, 3>(), 2) = z;
    _data(Fastor::fseq<0, 3>(), 3) = t;

    if (get_inverse) {
      _data_inv = Fastor::inverse(_data);
    }
  }

  /** Contructor with arguments: t, z, x
   *
   * @param t the translation (or origin of the new frame)
   * @param z the z axis of the new frame, normal vector for planes
   * @param x the x axis of the new frame
   *
   **/
  ALGEBRA_HOST
  transform3(const vector3 &t, const vector3 &z, const vector3 &x,
             bool get_inverse = true)
      : transform3(t, x, Fastor::cross(z, x), z, get_inverse) {}

  /** Constructor with arguments: translation
   *
   * @param t is the translation
   **/
  ALGEBRA_HOST
  transform3(const vector3 &t) {

    // The matrix needs to be initialized to the identity matrix first. In this
    // case, the `transform3` requires `_data` to look just like an identity
    // matrix except for the third column, which is the one we are modifying
    // here.
    _data.eye2();

    _data(Fastor::fseq<0, 3>(), 3) = t;

    _data_inv = Fastor::inverse(_data);
  }

  /** Constructor with arguments: matrix
   *
   * @param m is the full 4x4 matrix
   **/
  ALGEBRA_HOST
  transform3(const matrix44 &m) {
    _data = m;

    _data_inv = Fastor::inverse(_data);
  }

  /** Constructor with arguments: matrix as Fastor::Tensor<scalar_t, 16> of
   * scalars
   *
   * @param ma is the full 4x4 matrix as a 16-element array
   **/
  ALGEBRA_HOST
  transform3(const array_type<16> &ma) {
    _data = ma;

    _data_inv = Fastor::inverse(_data);
  }

  /** Default contructors */
  transform3() = default;
  transform3(const transform3 &rhs) = default;
  ~transform3() = default;

  /** Equality operator */
  ALGEBRA_HOST
  inline bool operator==(const transform3 &rhs) const {

    return Fastor::isequal(_data, rhs._data);
  }

  /** This method retrieves the rotation of a transform */
  ALGEBRA_HOST
  inline auto rotation() const {

    return Fastor::Tensor<scalar_t, 3, 3>(
        _data(Fastor::fseq<0, 3>(), Fastor::fseq<0, 3>()));
  }

  /** This method retrieves x axis */
  ALGEBRA_HOST_DEVICE
  inline point3 x() const { return _data(Fastor::fseq<0, 3>(), 0); }

  /** This method retrieves y axis */
  ALGEBRA_HOST_DEVICE
  inline point3 y() const { return _data(Fastor::fseq<0, 3>(), 1); }

  /** This method retrieves z axis */
  ALGEBRA_HOST_DEVICE
  inline point3 z() const { return _data(Fastor::fseq<0, 3>(), 2); }

  /** This method retrieves the translation of a transform */
  ALGEBRA_HOST
  inline vector3 translation() const { return _data(Fastor::fseq<0, 3>(), 3); }

  /** This method retrieves the 4x4 matrix of a transform */
  ALGEBRA_HOST
  inline matrix44 matrix() const { return _data; }

  /** This method retrieves the 4x4 matrix of an inverse transform */
  ALGEBRA_HOST
  inline matrix44 matrix_inverse() const { return _data_inv; }

  /** This method transform from a point from the local 3D cartesian frame to
   * the global 3D cartesian frame */
  ALGEBRA_HOST
  inline const point3 point_to_global(const point3 &v) const {

    Fastor::Tensor<scalar_type, 4> vector_4;
    vector_4(Fastor::fseq<0, 3>()) = v;
    vector_4[3] = static_cast<scalar_type>(1);
    return Fastor::Tensor<scalar_type, 3>(
        Fastor::matmul(_data, vector_4)(Fastor::fseq<0, 3>()));
  }

  /** This method transform from a vector from the global 3D cartesian frame
   * into the local 3D cartesian frame */
  ALGEBRA_HOST
  inline const point3 point_to_local(const point3 &v) const {

    Fastor::Tensor<scalar_type, 4> vector_4;
    vector_4(Fastor::fseq<0, 3>()) = v;
    vector_4[3] = static_cast<scalar_type>(1);
    return Fastor::Tensor<scalar_type, 3>(
        Fastor::matmul(_data_inv, vector_4)(Fastor::fseq<0, 3>()));
  }

  /** This method transform from a vector from the local 3D cartesian frame to
   * the global 3D cartesian frame */
  ALGEBRA_HOST
  inline const point3 vector_to_global(const vector3 &v) const {

    Fastor::Tensor<scalar_type, 4> vector_4;
    vector_4(Fastor::fseq<0, 3>()) = v;
    vector_4[3] = static_cast<scalar_type>(0);
    return Fastor::Tensor<scalar_type, 3>(
        Fastor::matmul(_data, vector_4)(Fastor::fseq<0, 3>()));
  }

  /** This method transform from a vector from the global 3D cartesian frame
   * into the local 3D cartesian frame */
  ALGEBRA_HOST
  inline const point3 vector_to_local(const vector3 &v) const {

    Fastor::Tensor<scalar_type, 4> vector_4;
    vector_4(Fastor::fseq<0, 3>()) = v;
    vector_4[3] = static_cast<scalar_type>(0);
    return Fastor::Tensor<scalar_type, 3>(
        Fastor::matmul(_data_inv, vector_4)(Fastor::fseq<0, 3>()));
  }
};  // struct transform3

}  // namespace algebra::fastor::math
