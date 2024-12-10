/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "test_base.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <array>
#include <climits>
#include <cmath>

/// Test case class, to be specialised for the different plugins
template <typename T>
class test_host_basics : public testing::Test, public test_base<T> {};
TYPED_TEST_SUITE_P(test_host_basics);

// This defines the local frame test suite
TYPED_TEST_P(test_host_basics, local_vectors) {

  // Construction
  typename TypeParam::point2 vA{0., 1.};
  ASSERT_EQ(vA[0], 0.);
  ASSERT_EQ(vA[1], 1.);

  // Assignment
  typename TypeParam::point2 vB = vA;
  ASSERT_EQ(vB[0], 0.);
  ASSERT_EQ(vB[1], 1.);

  // Addition
  typename TypeParam::point2 vC = vA + vB;
  ASSERT_EQ(vC[0], 0.);
  ASSERT_EQ(vC[1], 2.);

  // Multiplication by scalar
  typename TypeParam::point2 vC2 = vC * 2.;
  ASSERT_EQ(vC2[0], 0.);
  ASSERT_EQ(vC2[1], 4.);

  // Cast operations to phi, theta, eta, perp
  typename TypeParam::vector2 vD{1., 1.};
  typename TypeParam::scalar phi = algebra::getter::phi(vD);
  ASSERT_NEAR(phi, M_PI_4, this->m_epsilon);

  typename TypeParam::scalar perp = algebra::getter::perp(vD);
  ASSERT_NEAR(perp, std::sqrt(2.), this->m_epsilon);

  typename TypeParam::scalar norm = algebra::getter::norm(vD);
  ASSERT_NEAR(norm, std::sqrt(2.), this->m_epsilon);

  typename TypeParam::vector2 vDnorm = algebra::vector::normalize(vD);
  ASSERT_NEAR(vDnorm[0], 1. / std::sqrt(2.), this->m_epsilon);
  ASSERT_NEAR(vDnorm[1], 1. / std::sqrt(2.), this->m_epsilon);
}

// This defines the vector3 test suite
TYPED_TEST_P(test_host_basics, vector3) {

  // Construction
  typename TypeParam::vector3 vA{0., 1., 2.};
  ASSERT_EQ(vA[0], 0.);
  ASSERT_EQ(vA[1], 1.);
  ASSERT_EQ(vA[2], 2.);

  // Assignment
  typename TypeParam::vector3 vB = vA;
  ASSERT_EQ(vB[0], 0.);
  ASSERT_EQ(vB[1], 1.);
  ASSERT_EQ(vB[2], 2.);

  // Addition
  typename TypeParam::vector3 vC = vA + vB;
  ASSERT_EQ(vC[0], 0.);
  ASSERT_EQ(vC[1], 2.);
  ASSERT_EQ(vC[2], 4.);

  // Multiplication by scalar
  typename TypeParam::vector3 vC2 = vC * 2.0;
  ASSERT_EQ(vC2[0], 0.);
  ASSERT_EQ(vC2[1], 4.);
  ASSERT_EQ(vC2[2], 8.);

  // Cast operations to phi, theta, eta, perp
  typename TypeParam::vector3 vD{1., 1., 1.};
  typename TypeParam::scalar phi = algebra::getter::phi(vD);
  ASSERT_NEAR(phi, M_PI_4, this->m_epsilon);

  typename TypeParam::scalar theta = algebra::getter::theta(vD);
  ASSERT_NEAR(theta, std::atan2(std::sqrt(2.), 1.), this->m_epsilon);

  typename TypeParam::scalar eta = algebra::getter::eta(vD);
  ASSERT_NEAR(eta, 0.65847891569137573, this->m_isclose);

  typename TypeParam::scalar perp = algebra::getter::perp(vD);
  ASSERT_NEAR(perp, std::sqrt(2.), this->m_epsilon);

  typename TypeParam::scalar norm = algebra::getter::norm(vD);
  ASSERT_NEAR(norm, std::sqrt(3.), this->m_epsilon);

  // Test on matrix - vector operations
  typename TypeParam::vector3 vE{1., 2., 3.};

  typename TypeParam::template matrix<2, 3> m23;

  algebra::getter::element(m23, 0, 0) = 1;
  algebra::getter::element(m23, 0, 1) = 2;
  algebra::getter::element(m23, 0, 2) = 3;
  algebra::getter::element(m23, 1, 0) = 4;
  algebra::getter::element(m23, 1, 1) = 5;
  algebra::getter::element(m23, 1, 2) = 6;

  typename TypeParam::vector2 v2 = m23 * vE;

  ASSERT_NEAR(v2[0], 14, this->m_epsilon);
  ASSERT_NEAR(v2[1], 32, this->m_epsilon);

  // Cross product on vector3 and matrix<3,1>
  typename TypeParam::template matrix<3, 1> vF;
  algebra::getter::element(vF, 0, 0) = 5;
  algebra::getter::element(vF, 1, 0) = 6;
  algebra::getter::element(vF, 2, 0) = 13;

  typename TypeParam::vector3 vG = algebra::vector::cross(vD, vF);
  ASSERT_NEAR(vG[0], 7, this->m_epsilon);
  ASSERT_NEAR(vG[1], -8, this->m_epsilon);
  ASSERT_NEAR(vG[2], 1, this->m_epsilon);

  // Dot product on vector3 and matrix<3,1>
  auto dot = algebra::vector::dot(vG, vF);
  ASSERT_NEAR(dot, 0, this->m_epsilon);
}

// Test generic access to a 6x4 matrix
TYPED_TEST_P(test_host_basics, matrix64) {

  // Create the matrix.
  static constexpr typename TypeParam::size_type ROWS = 6;
  static constexpr typename TypeParam::size_type COLS = 4;
  typename TypeParam::template matrix<ROWS, COLS> m;

  // Fill it.
  for (typename TypeParam::size_type i = 0; i < ROWS; ++i) {
    for (typename TypeParam::size_type j = 0; j < COLS; ++j) {
      algebra::getter::element(m, i, j) =
          0.5f * static_cast<typename TypeParam::scalar>(i + j);
    }
  }

  // Check its content.
  const typename TypeParam::template matrix<ROWS, COLS>& m_const_ref = m;
  for (typename TypeParam::size_type i = 0; i < ROWS; ++i) {
    for (typename TypeParam::size_type j = 0; j < COLS; ++j) {
      const typename TypeParam::scalar ref =
          0.5f * static_cast<typename TypeParam::scalar>(i + j);
      ASSERT_NEAR(algebra::getter::element(m, i, j), ref, this->m_epsilon);
      ASSERT_NEAR(algebra::getter::element(m_const_ref, i, j), ref,
                  this->m_epsilon);
    }
  }

  // Test set_zero
  typename TypeParam::matrix_actor().set_zero(m);
  for (typename TypeParam::size_type i = 0; i < ROWS; ++i) {
    for (typename TypeParam::size_type j = 0; j < COLS; ++j) {
      ASSERT_NEAR(algebra::getter::element(m, i, j), 0., this->m_epsilon);
    }
  }

  // Test set_identity
  typename TypeParam::matrix_actor().set_identity(m);
  for (typename TypeParam::size_type i = 0; i < ROWS; ++i) {
    for (typename TypeParam::size_type j = 0; j < COLS; ++j) {
      if (i == j) {
        ASSERT_NEAR(algebra::getter::element(m, i, j), 1., this->m_epsilon);
      } else {
        ASSERT_NEAR(algebra::getter::element(m, i, j), 0., this->m_epsilon);
      }
    }
  }

  // Test block operations
  auto b13 = typename TypeParam::matrix_actor().template block<1, 3>(m, 0, 0);
  ASSERT_NEAR(algebra::getter::element(b13, 0, 0), 1., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(b13, 0, 1), 0., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(b13, 0, 2), 0., this->m_epsilon);

  auto b13_tp = typename TypeParam::matrix_actor().transpose(b13);
  ASSERT_NEAR(algebra::getter::element(b13_tp, 0, 0), 1., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(b13_tp, 1, 0), 0., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(b13_tp, 2, 0), 0., this->m_epsilon);

  auto b32 = typename TypeParam::matrix_actor().template block<3, 2>(m, 2, 2);
  ASSERT_NEAR(algebra::getter::element(b32, 0, 0), 1., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(b32, 0, 1), 0., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(b32, 1, 0), 0., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(b32, 1, 1), 1., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(b32, 2, 0), 0., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(b32, 2, 1), 0., this->m_epsilon);

  algebra::getter::element(b32, 0, 0) = 4;
  algebra::getter::element(b32, 0, 1) = 3;
  algebra::getter::element(b32, 1, 0) = 12;
  algebra::getter::element(b32, 1, 1) = 13;
  algebra::getter::element(b32, 2, 0) = 5;
  algebra::getter::element(b32, 2, 1) = 6;

  typename TypeParam::matrix_actor().set_block(m, b32, 2, 2);
  ASSERT_NEAR(algebra::getter::element(m, 2, 2), 4., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m, 2, 3), 3., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m, 3, 2), 12., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m, 3, 3), 13., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m, 4, 2), 5., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m, 4, 3), 6., this->m_epsilon);

  typename TypeParam::vector3 v = {10, 20, 30};
  typename TypeParam::matrix_actor().set_block(m, v, 0, 2);
  ASSERT_NEAR(algebra::getter::element(m, 0, 2), 10., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m, 1, 2), 20., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m, 2, 2), 30., this->m_epsilon);

  typename TypeParam::template matrix<3, 3> m33;
  algebra::getter::element(m33, 0, 0) = 1;
  algebra::getter::element(m33, 1, 0) = 2;
  algebra::getter::element(m33, 2, 0) = 3;
  algebra::getter::element(m33, 0, 1) = 5;
  algebra::getter::element(m33, 1, 1) = 6;
  algebra::getter::element(m33, 2, 1) = 7;
  algebra::getter::element(m33, 0, 2) = 9;
  algebra::getter::element(m33, 1, 2) = 10;
  algebra::getter::element(m33, 2, 2) = 11;

  const typename TypeParam::vector3 v2 = m33 * v;
  ASSERT_NEAR(v2[0], 380., this->m_epsilon);
  ASSERT_NEAR(v2[1], 440., this->m_epsilon);
  ASSERT_NEAR(v2[2], 500., this->m_epsilon);
}

// Test matrix operations with 3x3 matrix
TYPED_TEST_P(test_host_basics, matrix22) {

  typename TypeParam::template matrix<2, 2> m22;
  algebra::getter::element(m22, 0, 0) = 4;
  algebra::getter::element(m22, 0, 1) = 3;
  algebra::getter::element(m22, 1, 0) = 12;
  algebra::getter::element(m22, 1, 1) = 13;

  // Test 2 X 2 matrix determinant
  auto m22_det = typename TypeParam::matrix_actor().determinant(m22);
  ASSERT_NEAR(m22_det, 16., this->m_isclose);

  // Test 2 X 2 matrix inverse
  auto m22_inv = typename TypeParam::matrix_actor().inverse(m22);
  ASSERT_NEAR(algebra::getter::element(m22_inv, 0, 0), 13 / 16.,
              this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m22_inv, 0, 1), -3 / 16.,
              this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m22_inv, 1, 0), -12 / 16.,
              this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m22_inv, 1, 1), 4 / 16.,
              this->m_isclose);

  typename TypeParam::template matrix<3, 3> m33;
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
  auto m33_det = typename TypeParam::matrix_actor().determinant(m33);
  ASSERT_NEAR(m33_det, 20., this->m_isclose);

  // Test 3 X 3 matrix inverse
  auto m33_inv = typename TypeParam::matrix_actor().inverse(m33);
  ASSERT_NEAR(algebra::getter::element(m33_inv, 0, 0), -3 / 20.,
              this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m33_inv, 0, 1), 11 / 20.,
              this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m33_inv, 0, 2), -5 / 20.,
              this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m33_inv, 1, 0), -15 / 20.,
              this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m33_inv, 1, 1), -5 / 20.,
              this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m33_inv, 1, 2), 15 / 20.,
              this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m33_inv, 2, 0), 14 / 20.,
              this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m33_inv, 2, 1), 2 / 20.,
              this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m33_inv, 2, 2), -10 / 20.,
              this->m_isclose);

  // Test Zero
  typename TypeParam::template matrix<2, 3> m23 =
      typename TypeParam::matrix_actor().template zero<2, 3>();
  algebra::getter::element(m23, 0, 0) += 2;
  algebra::getter::element(m23, 0, 1) += 3;
  algebra::getter::element(m23, 0, 2) += 4;
  algebra::getter::element(m23, 1, 0) += 5;
  algebra::getter::element(m23, 1, 1) += 6;
  algebra::getter::element(m23, 1, 2) += 7;

  // Test scalar X Matrix
  m23 = 2. * m23;
  ASSERT_NEAR(algebra::getter::element(m23, 0, 0), 4., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m23, 0, 1), 6., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m23, 0, 2), 8., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m23, 1, 0), 10., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m23, 1, 1), 12., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m23, 1, 2), 14., this->m_epsilon);

  // Test Transpose
  auto m32 = typename TypeParam::matrix_actor().transpose(m23);

  // Test Identity and (Matrix + Matrix)
  m32 = m32 + typename TypeParam::matrix_actor().template identity<3, 2>();

  // Test Matrix X scalar
  m32 = m32 * 2.;

  ASSERT_NEAR(algebra::getter::element(m32, 0, 0), 10., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m32, 0, 1), 20., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m32, 1, 0), 12., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m32, 1, 1), 26., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m32, 2, 0), 16., this->m_epsilon);
  ASSERT_NEAR(algebra::getter::element(m32, 2, 1), 28., this->m_epsilon);

  // Test Matrix multiplication
  m22 = m22_inv * m23 * m33_inv * m32;

  ASSERT_NEAR(algebra::getter::element(m22, 0, 0), 6.225, this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m22, 0, 1), 14.675, this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m22, 1, 0), -3.3, this->m_isclose);
  ASSERT_NEAR(algebra::getter::element(m22, 1, 1), -7.9, this->m_isclose);

  // Test 6 X 6 big matrix determinant
  typename TypeParam::template matrix<6, 6> m66_big;
  algebra::getter::element(m66_big, 0, 0) = 1;
  algebra::getter::element(m66_big, 0, 1) = 0;
  algebra::getter::element(m66_big, 0, 2) = 3;
  algebra::getter::element(m66_big, 0, 3) = 0;
  algebra::getter::element(m66_big, 0, 4) = 0;
  algebra::getter::element(m66_big, 0, 5) = 0;

  algebra::getter::element(m66_big, 1, 0) = 0;
  algebra::getter::element(m66_big, 1, 1) = -2;
  algebra::getter::element(m66_big, 1, 2) = 4;
  algebra::getter::element(m66_big, 1, 3) = 0;
  algebra::getter::element(m66_big, 1, 4) = 5;
  algebra::getter::element(m66_big, 1, 5) = 0;

  algebra::getter::element(m66_big, 2, 0) = 0;
  algebra::getter::element(m66_big, 2, 1) = 0;
  algebra::getter::element(m66_big, 2, 2) = 3;
  algebra::getter::element(m66_big, 2, 3) = 0;
  algebra::getter::element(m66_big, 2, 4) = 0;
  algebra::getter::element(m66_big, 2, 5) = 0;

  algebra::getter::element(m66_big, 3, 0) = 0;
  algebra::getter::element(m66_big, 3, 1) = 0;
  algebra::getter::element(m66_big, 3, 2) = 0;
  algebra::getter::element(m66_big, 3, 3) = 4;
  algebra::getter::element(m66_big, 3, 4) = 0;
  algebra::getter::element(m66_big, 3, 5) = 0;

  algebra::getter::element(m66_big, 4, 0) = 0;
  algebra::getter::element(m66_big, 4, 1) = 0;
  algebra::getter::element(m66_big, 4, 2) = 0;
  algebra::getter::element(m66_big, 4, 3) = 0;
  algebra::getter::element(m66_big, 4, 4) = 9;
  algebra::getter::element(m66_big, 4, 5) = 0;

  algebra::getter::element(m66_big, 5, 0) = -1;
  algebra::getter::element(m66_big, 5, 1) = -1;
  algebra::getter::element(m66_big, 5, 2) = -1;
  algebra::getter::element(m66_big, 5, 3) = -1;
  algebra::getter::element(m66_big, 5, 4) = -1;
  algebra::getter::element(m66_big, 5, 5) = -1;

  auto m66_big_det = typename TypeParam::matrix_actor().determinant(m66_big);
  ASSERT_NEAR(m66_big_det, 216, 2 * this->m_isclose);

  // Test 6 X 6 small matrix determinant
  typename TypeParam::template matrix<6, 6> m66_small;

  algebra::getter::element(m66_small, 0, 0) =
      static_cast<typename TypeParam::scalar>(10.792386);
  algebra::getter::element(m66_small, 0, 1) =
      static_cast<typename TypeParam::scalar>(0.216181);
  algebra::getter::element(m66_small, 0, 2) =
      static_cast<typename TypeParam::scalar>(0.057650);
  algebra::getter::element(m66_small, 0, 3) =
      static_cast<typename TypeParam::scalar>(-0.002764);
  algebra::getter::element(m66_small, 0, 4) =
      static_cast<typename TypeParam::scalar>(0.000001);
  algebra::getter::element(m66_small, 0, 5) =
      static_cast<typename TypeParam::scalar>(0);

  algebra::getter::element(m66_small, 1, 0) =
      static_cast<typename TypeParam::scalar>(43.909368);
  algebra::getter::element(m66_small, 1, 1) =
      static_cast<typename TypeParam::scalar>(10.372997);
  algebra::getter::element(m66_small, 1, 2) =
      static_cast<typename TypeParam::scalar>(0.231496);
  algebra::getter::element(m66_small, 1, 3) =
      static_cast<typename TypeParam::scalar>(-0.065972);
  algebra::getter::element(m66_small, 1, 4) =
      static_cast<typename TypeParam::scalar>(-0.000002);
  algebra::getter::element(m66_small, 1, 5) =
      static_cast<typename TypeParam::scalar>(0);

  algebra::getter::element(m66_small, 2, 0) =
      static_cast<typename TypeParam::scalar>(0.045474);
  algebra::getter::element(m66_small, 2, 1) =
      static_cast<typename TypeParam::scalar>(-0.001730);
  algebra::getter::element(m66_small, 2, 2) =
      static_cast<typename TypeParam::scalar>(0.000246);
  algebra::getter::element(m66_small, 2, 3) =
      static_cast<typename TypeParam::scalar>(0.000004);
  algebra::getter::element(m66_small, 2, 4) =
      static_cast<typename TypeParam::scalar>(0);
  algebra::getter::element(m66_small, 2, 5) =
      static_cast<typename TypeParam::scalar>(0);

  algebra::getter::element(m66_small, 3, 0) =
      static_cast<typename TypeParam::scalar>(-0.255134);
  algebra::getter::element(m66_small, 3, 1) =
      static_cast<typename TypeParam::scalar>(-0.059846);
  algebra::getter::element(m66_small, 3, 2) =
      static_cast<typename TypeParam::scalar>(-0.001345);
  algebra::getter::element(m66_small, 3, 3) =
      static_cast<typename TypeParam::scalar>(0.000383);
  algebra::getter::element(m66_small, 3, 4) =
      static_cast<typename TypeParam::scalar>(0);
  algebra::getter::element(m66_small, 3, 5) =
      static_cast<typename TypeParam::scalar>(0);

  algebra::getter::element(m66_small, 4, 0) =
      static_cast<typename TypeParam::scalar>(-0.001490);
  algebra::getter::element(m66_small, 4, 1) =
      static_cast<typename TypeParam::scalar>(-0.000057);
  algebra::getter::element(m66_small, 4, 2) =
      static_cast<typename TypeParam::scalar>(-0.000008);
  algebra::getter::element(m66_small, 4, 3) =
      static_cast<typename TypeParam::scalar>(0.000001);
  algebra::getter::element(m66_small, 4, 4) =
      static_cast<typename TypeParam::scalar>(0.000001);
  algebra::getter::element(m66_small, 4, 5) =
      static_cast<typename TypeParam::scalar>(0);

  algebra::getter::element(m66_small, 5, 0) =
      static_cast<typename TypeParam::scalar>(0);
  algebra::getter::element(m66_small, 5, 1) =
      static_cast<typename TypeParam::scalar>(0);
  algebra::getter::element(m66_small, 5, 2) =
      static_cast<typename TypeParam::scalar>(0);
  algebra::getter::element(m66_small, 5, 3) =
      static_cast<typename TypeParam::scalar>(0);
  algebra::getter::element(m66_small, 5, 4) =
      static_cast<typename TypeParam::scalar>(0);
  algebra::getter::element(m66_small, 5, 5) =
      static_cast<typename TypeParam::scalar>(89875.517874);

  auto m66_small_det =
      typename TypeParam::matrix_actor().determinant(m66_small);
  ASSERT_NEAR((m66_small_det - 4.30636e-11) / 4.30636e-11, 0,
              2 * this->m_isclose);
}

// This defines the vector operation test suite
TYPED_TEST_P(test_host_basics, getter) {

  typename TypeParam::vector3 v3{1., 1., 1.};

  // Normalization
  typename TypeParam::vector3 v3n = algebra::vector::normalize(v3);
  ASSERT_NEAR(v3n[0], 1. / std::sqrt(3.), this->m_epsilon);
  ASSERT_NEAR(v3n[1], 1. / std::sqrt(3.), this->m_epsilon);
  ASSERT_NEAR(v3n[2], 1. / std::sqrt(3.), this->m_epsilon);

  // Cross product
  typename TypeParam::vector3 z =
      algebra::vector::normalize(typename TypeParam::vector3{3., 2., 1.});
  typename TypeParam::vector3 x =
      algebra::vector::normalize(typename TypeParam::vector3{2., -3., 0.});
  typename TypeParam::vector3 y =
      algebra::vector::cross(z, algebra::vector::normalize(x));

  // Check with dot product
  ASSERT_NEAR(algebra::vector::dot(x, y), 0., this->m_epsilon);
  ASSERT_NEAR(algebra::vector::dot(y, z), 0., this->m_epsilon);
  ASSERT_NEAR(algebra::vector::dot(z, x), 0., this->m_epsilon);
}

// This defines the transform3 test suite
TYPED_TEST_P(test_host_basics, transform3) {

  // Preparation work
  typename TypeParam::vector3 z =
      algebra::vector::normalize(typename TypeParam::vector3{3., 2., 1.});
  typename TypeParam::vector3 x =
      algebra::vector::normalize(typename TypeParam::vector3{2., -3., 0.});
  typename TypeParam::vector3 y = algebra::vector::cross(z, x);
  typename TypeParam::point3 t = {2., 3., 4.};

  // Test constructor from t, z, x
  typename TypeParam::transform3 trf1(t, z, x);
  ASSERT_TRUE(trf1 == trf1);
  typename TypeParam::transform3 trf2;
  trf2 = trf1;

  // Helper object for the matrix checks.
  auto element_getter = typename TypeParam::transform3::element_getter();

  const auto rot = trf2.rotation();
  ASSERT_NEAR(element_getter(rot, 0, 0), x[0], this->m_epsilon);
  ASSERT_NEAR(element_getter(rot, 1, 0), x[1], this->m_epsilon);
  ASSERT_NEAR(element_getter(rot, 2, 0), x[2], this->m_epsilon);
  ASSERT_NEAR(element_getter(rot, 0, 1), y[0], this->m_epsilon);
  ASSERT_NEAR(element_getter(rot, 1, 1), y[1], this->m_epsilon);
  ASSERT_NEAR(element_getter(rot, 2, 1), y[2], this->m_epsilon);
  ASSERT_NEAR(element_getter(rot, 0, 2), z[0], this->m_epsilon);
  ASSERT_NEAR(element_getter(rot, 1, 2), z[1], this->m_epsilon);
  ASSERT_NEAR(element_getter(rot, 2, 2), z[2], this->m_epsilon);

  auto trn = trf2.translation();
  ASSERT_NEAR(trn[0], 2., this->m_epsilon);
  ASSERT_NEAR(trn[1], 3., this->m_epsilon);
  ASSERT_NEAR(trn[2], 4., this->m_epsilon);

  // Test constructor from matrix
  auto m44 = trf2.matrix();
  typename TypeParam::transform3 trfm(m44);

  // Make sure that algebra::getter:vector can be called.
  [[maybe_unused]] typename TypeParam::vector3
      test_vector;  // we need to declare a variable in order to use the
                    // [[maybe_unused]] attribute here

  test_vector = algebra::getter::vector<3>(m44, 0, 0);
  test_vector = algebra::getter::vector<3>(m44, 0, 1);
  test_vector = algebra::getter::vector<3>(m44, 0, 2);

  // Test constructor from inverse matrix
  auto m44_inv = trf2.matrix_inverse();

  // Make sure that algebra::getter:vector can be called.
  test_vector = algebra::getter::vector<3>(m44_inv, 0, 0);
  test_vector = algebra::getter::vector<3>(m44_inv, 0, 1);
  test_vector = algebra::getter::vector<3>(m44_inv, 0, 2);

  // Re-evaluate rot and trn
  auto rotm = trfm.rotation();
  ASSERT_NEAR(element_getter(rotm, 0, 0), x[0], this->m_epsilon);
  ASSERT_NEAR(element_getter(rotm, 1, 0), x[1], this->m_epsilon);
  ASSERT_NEAR(element_getter(rotm, 2, 0), x[2], this->m_epsilon);
  ASSERT_NEAR(element_getter(rotm, 0, 1), y[0], this->m_epsilon);
  ASSERT_NEAR(element_getter(rotm, 1, 1), y[1], this->m_epsilon);
  ASSERT_NEAR(element_getter(rotm, 2, 1), y[2], this->m_epsilon);
  ASSERT_NEAR(element_getter(rotm, 0, 2), z[0], this->m_epsilon);
  ASSERT_NEAR(element_getter(rotm, 1, 2), z[1], this->m_epsilon);
  ASSERT_NEAR(element_getter(rotm, 2, 2), z[2], this->m_epsilon);

  auto trnm = trfm.translation();
  ASSERT_NEAR(trnm[0], 2., this->m_epsilon);
  ASSERT_NEAR(trnm[1], 3., this->m_epsilon);
  ASSERT_NEAR(trnm[2], 4., this->m_epsilon);

  // Check a contruction from an array[16]
  std::array<typename TypeParam::scalar, 16> matray_helper = {
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
  typename TypeParam::transform3::template array_type<16> matray;
  for (unsigned int i = 0u; i < 16u; ++i) {
    matray[i] = matray_helper[i];
  }
  typename TypeParam::transform3 trfma(matray);

  // Re-evaluate rot and trn
  auto rotma = trfma.rotation();
  ASSERT_NEAR(element_getter(rotma, 0, 0), 1., this->m_epsilon);
  ASSERT_NEAR(element_getter(rotma, 1, 0), 0., this->m_epsilon);
  ASSERT_NEAR(element_getter(rotma, 2, 0), 0., this->m_epsilon);
  ASSERT_NEAR(element_getter(rotma, 0, 1), 0., this->m_epsilon);
  ASSERT_NEAR(element_getter(rotma, 1, 1), 1., this->m_epsilon);
  ASSERT_NEAR(element_getter(rotma, 2, 1), 0., this->m_epsilon);
  ASSERT_NEAR(element_getter(rotma, 0, 2), 0., this->m_epsilon);
  ASSERT_NEAR(element_getter(rotma, 1, 2), 0., this->m_epsilon);
  ASSERT_NEAR(element_getter(rotma, 2, 2), 1., this->m_epsilon);

  auto trnma = trfma.translation();
  ASSERT_NEAR(trnma[0], 0., this->m_epsilon);
  ASSERT_NEAR(trnma[1], 0., this->m_epsilon);
  ASSERT_NEAR(trnma[2], 0., this->m_epsilon);
}

// This test global coordinate transforms
TYPED_TEST_P(test_host_basics, global_transformations) {

  // Preparation work
  typename TypeParam::vector3 z =
      algebra::vector::normalize(typename TypeParam::vector3{3., 2., 1.});
  typename TypeParam::vector3 x =
      algebra::vector::normalize(typename TypeParam::vector3{2., -3., 0.});
  [[maybe_unused]] typename TypeParam::vector3 y = algebra::vector::cross(z, x);
  typename TypeParam::point3 t = {2., 3., 4.};
  typename TypeParam::transform3 trf(t, z, x);

  // Check that local origin translates into global translation
  typename TypeParam::point3 lzero = {0., 0., 0.};
  typename TypeParam::point3 gzero = trf.point_to_global(lzero);
  ASSERT_NEAR(gzero[0], t[0], this->m_epsilon);
  ASSERT_NEAR(gzero[1], t[1], this->m_epsilon);
  ASSERT_NEAR(gzero[2], t[2], this->m_epsilon);

  // Check a round trip for point
  typename TypeParam::point3 lpoint = {3., 4., 5.};
  typename TypeParam::point3 gpoint = trf.point_to_global(lpoint);
  typename TypeParam::point3 lpoint_r = trf.point_to_local(gpoint);
  ASSERT_NEAR(lpoint[0], lpoint_r[0], this->m_isclose);
  ASSERT_NEAR(lpoint[1], lpoint_r[1], this->m_isclose);
  ASSERT_NEAR(lpoint[2], lpoint_r[2], this->m_isclose);

  // Check a point versus vector transform
  // vector should not change if transformed by a pure translation
  typename TypeParam::transform3 ttrf(t);

  typename TypeParam::vector3 gvector = {1., 1., 1.};
  typename TypeParam::vector3 lvector = ttrf.vector_to_local(gvector);
  ASSERT_NEAR(gvector[0], lvector[0], this->m_isclose);
  ASSERT_NEAR(gvector[1], lvector[1], this->m_isclose);
  ASSERT_NEAR(gvector[2], lvector[2], this->m_isclose);

  // Check a round trip for vector
  typename TypeParam::vector3 lvectorB = {7., 8., 9.};
  typename TypeParam::vector3 gvectorB = trf.vector_to_local(lvectorB);
  typename TypeParam::vector3 lvectorC = trf.vector_to_global(gvectorB);
  ASSERT_NEAR(lvectorB[0], lvectorC[0], this->m_isclose);
  ASSERT_NEAR(lvectorB[1], lvectorC[1], this->m_isclose);
  ASSERT_NEAR(lvectorB[2], lvectorC[2], this->m_isclose);
}

REGISTER_TYPED_TEST_SUITE_P(test_host_basics, local_vectors, vector3, matrix64,
                            matrix22, getter, transform3,
                            global_transformations);
