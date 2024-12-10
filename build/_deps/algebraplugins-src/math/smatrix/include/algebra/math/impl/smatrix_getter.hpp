/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/qualifiers.hpp"

// ROOT/Smatrix include(s).
#include <Math/Expression.h>
#include <Math/Functions.h>
#include <Math/SMatrix.h>
#include <Math/SVector.h>
#include <TMath.h>

// System include(s).
#include <cassert>

namespace algebra::smatrix::math {

/** This method retrieves phi from a vector, vector base with rows >= 2
 *
 * @param v the input vector
 **/
template <typename scalar_t, auto N, std::enable_if_t<N >= 2, bool> = true>
ALGEBRA_HOST inline scalar_t phi(
    const ROOT::Math::SVector<scalar_t, N> &v) noexcept {

  return static_cast<scalar_t>(TMath::ATan2(v[1], v[0]));
}

template <typename scalar_t, class A, auto N,
          std::enable_if_t<N >= 2, bool> = true>
ALGEBRA_HOST inline scalar_t phi(
    const ROOT::Math::VecExpr<A, scalar_t, N> &v) noexcept {

  return static_cast<scalar_t>(TMath::ATan2(v.apply(1), v.apply(0)));
}

/** This method retrieves theta from a vector, vector base with rows >= 3
 *
 * @param v the input vector
 **/
template <typename scalar_t, auto N, std::enable_if_t<N >= 3, bool> = true>
ALGEBRA_HOST inline scalar_t theta(
    const ROOT::Math::SVector<scalar_t, N> &v) noexcept {

  return static_cast<scalar_t>(
      TMath::ATan2(TMath::Sqrt(v[0] * v[0] + v[1] * v[1]), v[2]));
}

template <typename scalar_t, class A, auto N,
          std::enable_if_t<N >= 3, bool> = true>
ALGEBRA_HOST inline scalar_t theta(
    const ROOT::Math::VecExpr<A, scalar_t, N> &v) noexcept {

  return static_cast<scalar_t>(TMath::ATan2(
      TMath::Sqrt(v.apply(0) * v.apply(0) + v.apply(1) * v.apply(1)),
      v.apply(2)));
}

/** This method retrieves the norm of a vector, no dimension restriction
 *
 * @param v the input vector
 **/
template <typename scalar_t, auto N>
ALGEBRA_HOST inline scalar_t norm(const ROOT::Math::SVector<scalar_t, N> &v) {

  return static_cast<scalar_t>(TMath::Sqrt(ROOT::Math::Dot(v, v)));
}

template <typename scalar_t, class A, auto N>
ALGEBRA_HOST inline scalar_t norm(
    const ROOT::Math::VecExpr<A, scalar_t, N> &v) {

  return static_cast<scalar_t>(TMath::Sqrt(ROOT::Math::Dot(v, v)));
}

/** This method retrieves the pseudo-rapidity from a vector or vector base with
 * rows >= 3
 *
 * @param v the input vector
 **/
template <typename scalar_t, auto N, std::enable_if_t<N >= 3, bool> = true>
ALGEBRA_HOST inline scalar_t eta(
    const ROOT::Math::SVector<scalar_t, N> &v) noexcept {

  return static_cast<scalar_t>(TMath::ATanH(v[2] / norm(v)));
}

template <typename scalar_t, class A, auto N,
          std::enable_if_t<N >= 3, bool> = true>
ALGEBRA_HOST inline scalar_t eta(
    const ROOT::Math::VecExpr<A, scalar_t, N> &v) noexcept {

  return static_cast<scalar_t>(TMath::ATanH(v.apply(2) / norm(v)));
}

/** This method retrieves the perpendicular magnitude of a vector with rows >= 2
 *
 * @param v the input vector
 **/
template <typename scalar_t, auto N, std::enable_if_t<N >= 2, bool> = true>
ALGEBRA_HOST inline scalar_t perp(
    const ROOT::Math::SVector<scalar_t, N> &v) noexcept {

  return static_cast<scalar_t>(TMath::Sqrt(v[0] * v[0] + v[1] * v[1]));
}

template <typename scalar_t, class A, auto N,
          std::enable_if_t<N >= 2, bool> = true>
ALGEBRA_HOST inline scalar_t perp(
    const ROOT::Math::VecExpr<A, scalar_t, N> &v) noexcept {

  return static_cast<scalar_t>(
      TMath::Sqrt(v.apply(0) * v.apply(0) + v.apply(1) * v.apply(1)));
}

/// Functor used to access elements of SMatrix matrices
template <typename scalar_t>
struct element_getter {

  template <unsigned int ROWS, unsigned int COLS>
  using matrix_type = ROOT::Math::SMatrix<scalar_t, ROWS, COLS>;

  template <unsigned int ROWS, unsigned int COLS>
  ALGEBRA_HOST_DEVICE inline scalar_t &operator()(matrix_type<ROWS, COLS> &m,
                                                  unsigned int row,
                                                  unsigned int col) const {

    assert(row < ROWS);
    assert(col < COLS);
    return m(row, col);
  }

  template <unsigned int ROWS, unsigned int COLS>
  ALGEBRA_HOST_DEVICE inline scalar_t operator()(
      const matrix_type<ROWS, COLS> &m, unsigned int row,
      unsigned int col) const {

    assert(row < ROWS);
    assert(col < COLS);
    return m(row, col);
  }
};  // element_getter

/// Function extracting an element from a matrix (const)
template <typename scalar_t, unsigned int ROWS, unsigned int COLS>
ALGEBRA_HOST_DEVICE inline scalar_t element(
    const ROOT::Math::SMatrix<scalar_t, ROWS, COLS> &m, unsigned int row,
    unsigned int col) {

  return element_getter<scalar_t>()(m, row, col);
}

/// Function extracting an element from a matrix (non-const)
template <typename scalar_t, unsigned int ROWS, unsigned int COLS>
ALGEBRA_HOST_DEVICE inline scalar_t &element(
    ROOT::Math::SMatrix<scalar_t, ROWS, COLS> &m, unsigned int row,
    unsigned int col) {

  return element_getter<scalar_t>()(m, row, col);
}

/// Functor used to extract a block from SMatrix matrices
template <typename scalar_t>
struct block_getter {

  template <unsigned int ROWS, unsigned int COLS>
  using matrix_type = ROOT::Math::SMatrix<scalar_t, ROWS, COLS>;

  template <unsigned int ROWS, unsigned int COLS, class input_matrix_type>
  ALGEBRA_HOST_DEVICE matrix_type<ROWS, COLS> operator()(
      const input_matrix_type &m, unsigned int row, unsigned int col) const {

    return m.template Sub<matrix_type<ROWS, COLS> >(row, col);
  }
};  // struct block_getter

}  // namespace algebra::smatrix::math
