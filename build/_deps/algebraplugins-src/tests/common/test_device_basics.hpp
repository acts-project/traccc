/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/qualifiers.hpp"

// Local include(s).
#include "test_base.hpp"

/// Operations that can be executed on the host or a device
///
/// This class does not execute any tests itself. It performs "operations",
/// whose results could be tested in code that uses this type.
///
template <typename T>
class test_device_basics : public test_base<T> {

 public:
  /// @name Type definitions
  /// @{

  using scalar = typename test_base<T>::scalar;
  using point2 = typename test_base<T>::point2;
  using point3 = typename test_base<T>::point3;
  using vector2 = typename test_base<T>::vector2;
  using vector3 = typename test_base<T>::vector3;
  using transform3 = typename test_base<T>::transform3;
  using size_type = typename test_base<T>::size_type;
  template <size_type ROWS, size_type COLS>
  using matrix = typename test_base<T>::template matrix<ROWS, COLS>;
  using matrix_actor = typename test_base<T>::matrix_actor;

  /// @}

  /// Perform various 2D vector operations, and produce a scalar output
  ALGEBRA_HOST_DEVICE
  scalar vector_2d_ops(point2 a, point2 b) const {

    point2 c = a + b;
    point2 c2 = c * 2.0;

    scalar phi = algebra::getter::phi(c2);
    scalar perp = algebra::getter::perp(c2);
    scalar norm1 = algebra::getter::norm(c2);

    scalar dot = algebra::vector::dot(a, b);
    point2 norm2 = algebra::vector::normalize(c);
    scalar norm3 = algebra::getter::norm(norm2);

    return (phi + perp + norm1 + dot + norm3);
  }

  /// Perform various 3D vector operations, and produce a scalar output
  ALGEBRA_HOST_DEVICE
  scalar vector_3d_ops(vector3 a, vector3 b) const {

    vector3 c = a + b;
    vector3 c2 = c * 2.0;

    scalar phi = algebra::getter::phi(c2);
    scalar perp = algebra::getter::perp(c2);
    scalar norm1 = algebra::getter::norm(c2);

    vector3 d = algebra::vector::cross(a, b);

    scalar dot = algebra::vector::dot(a, d);
    vector3 norm2 = algebra::vector::normalize(c);
    scalar norm3 = algebra::getter::norm(norm2);

    return (phi + perp + norm1 + dot + norm3);
  }

  /// Perform some trivial operations on an asymmetrix matrix
  ALGEBRA_HOST_DEVICE
  scalar matrix64_ops(const matrix<6, 4>& m) const {

    matrix<6, 4> m2;
    for (size_type i = 0; i < 6; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        algebra::getter::element(m2, i, j) = algebra::getter::element(m, i, j);
      }
    }

    scalar result = 0.;
    for (size_type i = 0; i < 6; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        result += 0.6f * algebra::getter::element(m, i, j) +
                  0.7f * algebra::getter::element(m2, i, j);
      }
    }

    // Test set_zero
    matrix_actor().set_zero(m2);
    for (size_type i = 0; i < 6; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        result += 0.4f * algebra::getter::element(m2, i, j);
      }
    }

    // Test set_identity
    matrix_actor().set_identity(m2);
    for (size_type i = 0; i < 6; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        result += 0.3f * algebra::getter::element(m2, i, j);
      }
    }

    // Test block operations
    auto b13 = matrix_actor().template block<1, 3>(m2, 0, 0);
    auto b13_tp = matrix_actor().transpose(b13);
    algebra::getter::element(b13_tp, 0, 0) = 1;
    algebra::getter::element(b13_tp, 1, 0) = 2;
    algebra::getter::element(b13_tp, 2, 0) = 3;
    matrix_actor().set_block(m2, b13_tp, 0, 0);

    auto b32 = matrix_actor().template block<3, 2>(m2, 2, 2);
    algebra::getter::element(b32, 0, 0) = 4;
    algebra::getter::element(b32, 0, 1) = 3;
    algebra::getter::element(b32, 1, 0) = 12;
    algebra::getter::element(b32, 1, 1) = 13;
    algebra::getter::element(b32, 2, 0) = 5;
    algebra::getter::element(b32, 2, 1) = 6;

    matrix_actor().set_block(m2, b32, 2, 2);
    for (size_type i = 0; i < 6; ++i) {
      for (size_type j = 0; j < 4; ++j) {
        result += 0.57f * algebra::getter::element(m2, i, j);
      }
    }

    return result;
  }

  /// Perform some trivial operations on an asymmetrix matrix
  ALGEBRA_HOST_DEVICE
  scalar matrix22_ops(const matrix<2, 2>& m22) const {

    // Test 2 X 2 matrix determinant
    auto m22_det = matrix_actor().determinant(m22);

    // Test 2 X 2 matrix inverse
    auto m22_inv = matrix_actor().inverse(m22);

    matrix<3, 3> m33;
    algebra::getter::element(m33, 0, 0) = 1;
    algebra::getter::element(m33, 0, 1) = 5;
    algebra::getter::element(m33, 0, 2) = 7;
    algebra::getter::element(m33, 1, 0) = 3;
    algebra::getter::element(m33, 1, 1) = 5;
    algebra::getter::element(m33, 1, 2) = 6;
    algebra::getter::element(m33, 2, 0) = 2;
    algebra::getter::element(m33, 2, 1) = 8;
    algebra::getter::element(m33, 2, 2) = 9;

    // Test 3 X 3 matrix determinant
    auto m33_det = matrix_actor().determinant(m33);

    // Test 3 X 3 matrix inverse
    auto m33_inv = matrix_actor().inverse(m33);

    // Test Zero
    auto m23 = matrix_actor().template zero<2, 3>();
    algebra::getter::element(m23, 0, 0) += 2;
    algebra::getter::element(m23, 0, 1) += 3;
    algebra::getter::element(m23, 0, 2) += 4;
    algebra::getter::element(m23, 1, 0) += 5;
    algebra::getter::element(m23, 1, 1) += 6;
    algebra::getter::element(m23, 1, 2) += 7;

    // Test scalar X Matrix
    m23 = 2. * m23;

    // Test Transpose
    auto m32 = matrix_actor().transpose(m23);

    // Test Identity and (Matrix + Matrix)
    m32 = m32 + matrix_actor().template identity<3, 2>();

    // Test Matrix X scalar
    m32 = m32 * 2.;

    // Test Matrix multiplication
    auto new_m22 = m22_inv * m23 * m33_inv * m32;

    scalar result = 0;
    result += m22_det;
    result += m33_det;
    result += algebra::getter::element(new_m22, 0, 0);
    result += algebra::getter::element(new_m22, 0, 1);
    result += algebra::getter::element(new_m22, 1, 0);
    result += algebra::getter::element(new_m22, 1, 1);

    return result;
  }

  /// Perform various operations using the @c transform3 type
  ALGEBRA_HOST_DEVICE
  scalar transform3_ops(vector3 t1, vector3 t2, vector3 t3, vector3 a,
                        vector3 b) const {

    transform3 tr1(t1, t2, t3);
    transform3 tr2;
    tr2 = tr1;

    point3 translation = tr2.translation();

    point3 gpoint = tr2.point_to_global(a);
    point3 lpoint = tr2.point_to_local(b);

    vector3 gvec = tr2.vector_to_global(a);
    vector3 lvec = tr2.vector_to_local(b);

    return {algebra::getter::norm(translation) + algebra::getter::perp(gpoint) +
            algebra::getter::phi(lpoint) + algebra::vector::dot(gvec, lvec)};
  }

};  // class test_device_basics
