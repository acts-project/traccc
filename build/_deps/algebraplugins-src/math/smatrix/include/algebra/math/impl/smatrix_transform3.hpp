/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/impl/smatrix_errorcheck.hpp"
#include "algebra/qualifiers.hpp"

// ROOT/Smatrix include(s).
#include "Math/SMatrix.h"
#include "Math/SVector.h"

namespace algebra::smatrix::math {

/** Transform wrapper class to ensure standard API within differnt plugins
 *
 **/
template <typename scalar_t, typename matrix_actor_t>
struct transform3 {

  /// @name Type definitions for the struct
  /// @{

  /// Array type used by the transform
  template <unsigned int N>
  using array_type = ROOT::Math::SVector<scalar_t, N>;
  /// Scalar type used by the transform
  using scalar_type = scalar_t;

  /// 3-element "vector" type
  using vector3 = array_type<3>;
  /// Point in 3D space
  using point3 = vector3;
  /// Point in 2D space
  using point2 = array_type<2>;

  /// 4x4 matrix type
  using matrix44 = ROOT::Math::SMatrix<scalar_type, 4, 4>;

  /// Function (object) used for accessing a matrix element
  using element_getter = algebra::smatrix::math::element_getter<scalar_t>;

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

  matrix44 _data = ROOT::Math::SMatrixIdentity();
  matrix44 _data_inv = ROOT::Math::SMatrixIdentity();

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
    _data(0, 0) = x[0];
    _data(1, 0) = x[1];
    _data(2, 0) = x[2];
    _data(0, 1) = y[0];
    _data(1, 1) = y[1];
    _data(2, 1) = y[2];
    _data(0, 2) = z[0];
    _data(1, 2) = z[1];
    _data(2, 2) = z[2];
    _data(0, 3) = t[0];
    _data(1, 3) = t[1];
    _data(2, 3) = t[2];

    if (get_inverse) {
      int ifail = 0;
      _data_inv = _data.Inverse(ifail);
      SMATRIX_CHECK(ifail);
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
      : transform3(t, x, ROOT::Math::Cross(z, x), z, get_inverse) {}

  /** Constructor with arguments: translation
   *
   * @param t is the translation
   **/
  ALGEBRA_HOST
  transform3(const vector3 &t) {

    _data(0, 3) = t[0];
    _data(1, 3) = t[1];
    _data(2, 3) = t[2];

    int ifail = 0;
    _data_inv = _data.Inverse(ifail);
    SMATRIX_CHECK(ifail);
  }

  /** Constructor with arguments: matrix
   *
   * @param m is the full 4x4 matrix
   **/
  ALGEBRA_HOST
  transform3(const matrix44 &m) {
    _data = m;

    int ifail = 0;
    _data_inv = _data.Inverse(ifail);
    SMATRIX_CHECK(ifail);
  }

  /** Constructor with arguments: matrix as ROOT::Math::SVector<scalar_t, 16> of
   * scalars
   *
   * @param ma is the full 4x4 matrix as a 16-element array
   **/
  ALGEBRA_HOST
  transform3(const array_type<16> &ma) {

    _data(0, 0) = ma[0];
    _data(1, 0) = ma[4];
    _data(2, 0) = ma[8];
    _data(3, 0) = ma[12];
    _data(0, 1) = ma[1];
    _data(1, 1) = ma[5];
    _data(2, 1) = ma[9];
    _data(3, 1) = ma[13];
    _data(0, 2) = ma[2];
    _data(1, 2) = ma[6];
    _data(2, 2) = ma[10];
    _data(3, 2) = ma[14];
    _data(0, 3) = ma[3];
    _data(1, 3) = ma[7];
    _data(2, 3) = ma[11];
    _data(3, 3) = ma[15];

    int ifail = 0;
    _data_inv = _data.Inverse(ifail);
    // Ignore failures here, since the unit test does manage to trigger an error
    // from ROOT in this place...
  }

  /** Default contructors */
  transform3() = default;
  transform3(const transform3 &rhs) = default;
  ~transform3() = default;

  /** Equality operator */
  ALGEBRA_HOST
  inline bool operator==(const transform3 &rhs) const {

    return _data == rhs._data;
  }

  /** This method retrieves the rotation of a transform */
  ALGEBRA_HOST
  inline auto rotation() const {

    return (_data.template Sub<ROOT::Math::SMatrix<scalar_type, 3, 3> >(0, 0));
  }

  /** This method retrieves x axis */
  ALGEBRA_HOST_DEVICE
  inline point3 x() const { return (_data.template SubCol<vector3>(0, 0)); }

  /** This method retrieves y axis */
  ALGEBRA_HOST_DEVICE
  inline point3 y() const { return (_data.template SubCol<vector3>(1, 0)); }

  /** This method retrieves z axis */
  ALGEBRA_HOST_DEVICE
  inline point3 z() const { return (_data.template SubCol<vector3>(2, 0)); }

  /** This method retrieves the translation of a transform */
  ALGEBRA_HOST
  inline vector3 translation() const {

    return (_data.template SubCol<vector3>(3, 0));
  }

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

    ROOT::Math::SVector<scalar_type, 4> vector_4;
    vector_4.Place_at(v, 0);
    vector_4[3] = static_cast<scalar_type>(1);
    return ROOT::Math::SVector<scalar_type, 4>(_data * vector_4)
        .template Sub<point3>(0);
  }

  /** This method transform from a vector from the global 3D cartesian frame
   * into the local 3D cartesian frame */
  ALGEBRA_HOST
  inline const point3 point_to_local(const point3 &v) const {

    ROOT::Math::SVector<scalar_type, 4> vector_4;
    vector_4.Place_at(v, 0);
    vector_4[3] = static_cast<scalar_type>(1);
    return ROOT::Math::SVector<scalar_type, 4>(_data_inv * vector_4)
        .template Sub<point3>(0);
  }

  /** This method transform from a vector from the local 3D cartesian frame to
   * the global 3D cartesian frame */
  ALGEBRA_HOST
  inline const point3 vector_to_global(const vector3 &v) const {

    ROOT::Math::SVector<scalar_type, 4> vector_4;
    vector_4.Place_at(v, 0);
    return ROOT::Math::SVector<scalar_type, 4>(_data * vector_4)
        .template Sub<point3>(0);
  }

  /** This method transform from a vector from the global 3D cartesian frame
   * into the local 3D cartesian frame */
  ALGEBRA_HOST
  inline const point3 vector_to_local(const vector3 &v) const {

    ROOT::Math::SVector<scalar_type, 4> vector_4;
    vector_4.Place_at(v, 0);
    return ROOT::Math::SVector<scalar_type, 4>(_data_inv * vector_4)
        .template Sub<point3>(0);
  }
};  // struct transform3

}  // namespace algebra::smatrix::math
