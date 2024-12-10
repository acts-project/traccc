/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/common.hpp"
#include "algebra/qualifiers.hpp"

namespace algebra::cmath {

/** Cross product between two input vectors - 3 Dim
 *
 * @tparam derived_type_lhs is the first matrix (epresseion) template
 * @tparam derived_type_rhs is the second matrix (epresseion) template
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return a vector (expression) representing the cross product
 **/
template <typename size_type, template <typename, size_type> class array_t,
          typename scalar_t, size_type N,
          std::enable_if_t<N >= 3 && std::is_scalar_v<scalar_t>, bool> = true>
ALGEBRA_HOST_DEVICE inline array_t<scalar_t, N> cross(
    const array_t<scalar_t, N> &a, const array_t<scalar_t, N> &b) {

  return {a[1] * b[2] - b[1] * a[2], a[2] * b[0] - b[2] * a[0],
          a[0] * b[1] - b[0] * a[1]};
}

/** Cross product between two input vectors - 3 Dim
 *
 * @tparam derived_type_lhs is the first matrix (epresseion) template
 * @tparam derived_type_rhs is the second matrix (epresseion) template
 *
 * @param a the first input vector
 * @param b the second input matrix with single column
 *
 * @return a vector (expression) representing the cross product
 **/
template <typename size_type, template <typename, size_type> class array_t,
          typename scalar_t, size_type N, size_type COLS,
          std::enable_if_t<N >= 3 && COLS == 1 && std::is_scalar_v<scalar_t>,
                           bool> = true>
ALGEBRA_HOST_DEVICE inline array_t<scalar_t, N> cross(
    const array_t<scalar_t, N> &a,
    const array_t<array_t<scalar_t, N>, COLS> &b) {

  return {a[1] * b[0][2] - b[0][1] * a[2], a[2] * b[0][0] - b[0][2] * a[0],
          a[0] * b[0][1] - b[0][0] * a[1]};
}

/** Cross product between two input vectors - 3 Dim
 *
 * @tparam derived_type_lhs is the first matrix (epresseion) template
 * @tparam derived_type_rhs is the second matrix (epresseion) template
 *
 * @param a the first input matrix with single column
 * @param b the second input vector
 *
 * @return a vector (expression) representing the cross product
 **/
template <typename size_type, template <typename, size_type> class array_t,
          typename scalar_t, size_type N, size_type COLS,
          std::enable_if_t<N >= 3 && COLS == 1 && std::is_scalar_v<scalar_t>,
                           bool> = true>
ALGEBRA_HOST_DEVICE inline array_t<scalar_t, N> cross(
    const array_t<array_t<scalar_t, N>, COLS> &a,
    const array_t<scalar_t, N> &b) {

  return {a[0][1] * b[2] - b[1] * a[0][2], a[0][2] * b[0] - b[2] * a[0][0],
          a[0][0] * b[1] - b[0] * a[0][1]};
}

/** Cross product between two input vectors - 3 Dim
 *
 * @tparam derived_type_lhs is the first matrix (epresseion) template
 * @tparam derived_type_rhs is the second matrix (epresseion) template
 *
 * @param a the first input matrix with single column
 * @param b the second input matrix with single column
 *
 * @return a vector (expression) representing the cross product
 **/
template <typename size_type, template <typename, size_type> class array_t,
          typename scalar_t, size_type N, size_type COLS,
          std::enable_if_t<N >= 3 && COLS == 1 && std::is_scalar_v<scalar_t>,
                           bool> = true>
ALGEBRA_HOST_DEVICE inline array_t<scalar_t, N> cross(
    const array_t<array_t<scalar_t, N>, COLS> &a,
    const array_t<array_t<scalar_t, N>, COLS> &b) {

  return {a[0][1] * b[0][2] - b[0][1] * a[0][2],
          a[0][2] * b[0][0] - b[0][2] * a[0][0],
          a[0][0] * b[0][1] - b[0][0] * a[0][1]};
}

/** Dot product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return the scalar dot product value
 **/
template <typename size_type, template <typename, size_type> class array_t,
          typename scalar_t, size_type N,
          std::enable_if_t<std::is_scalar_v<scalar_t>, bool> = true>
ALGEBRA_HOST_DEVICE inline scalar_t dot(const array_t<scalar_t, N> &a,
                                        const array_t<scalar_t, N> &b) {
  scalar_t ret = 0;
  for (size_type i = 0; i < N; i++) {
    ret += a[i] * b[i];
  }
  return ret;
}

/** Dot product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input matrix with single column
 *
 * @return the scalar dot product value
 **/
template <
    typename size_type, template <typename, size_type> class array_t,
    typename scalar_t, size_type N, size_type COLS,
    std::enable_if_t<COLS == 1 && std::is_scalar_v<scalar_t>, bool> = true>
ALGEBRA_HOST_DEVICE inline scalar_t dot(
    const array_t<scalar_t, N> &a,
    const array_t<array_t<scalar_t, N>, COLS> &b) {
  scalar_t ret = 0;
  for (size_type i = 0; i < N; i++) {
    ret += a[i] * b[0][i];
  }
  return ret;
}

/** Dot product between two input vectors
 *
 * @param a the first input matrix with single column
 * @param b the second input vector
 *
 * @return the scalar dot product value
 **/
template <
    typename size_type, template <typename, size_type> class array_t,
    typename scalar_t, size_type N, size_type COLS,
    std::enable_if_t<COLS == 1 && std::is_scalar_v<scalar_t>, bool> = true>
ALGEBRA_HOST_DEVICE inline scalar_t dot(
    const array_t<array_t<scalar_t, N>, COLS> &a,
    const array_t<scalar_t, N> &b) {
  scalar_t ret = 0;
  for (size_type i = 0; i < N; i++) {
    ret += a[0][i] * b[i];
  }
  return ret;
}

/** Dot product between two input vectors
 *
 * @param a the first input matrix with single column
 * @param b the second input matrix with single column
 *
 * @return the scalar dot product value
 **/
template <
    typename size_type, template <typename, size_type> class array_t,
    typename scalar_t, size_type N, size_type COLS,
    std::enable_if_t<COLS == 1 && std::is_scalar_v<scalar_t>, bool> = true>
ALGEBRA_HOST_DEVICE inline scalar_t dot(
    const array_t<array_t<scalar_t, N>, COLS> &a,
    const array_t<array_t<scalar_t, N>, COLS> &b) {
  scalar_t ret = 0;
  for (size_type i = 0; i < N; i++) {
    ret += a[0][i] * b[0][i];
  }
  return ret;
}

/** Get a normalized version of the input vector
 *
 * @param v the input vector
 **/
template <typename size_type, template <typename, size_type> class array_t,
          typename scalar_t, size_type N,
          std::enable_if_t<std::is_scalar_v<scalar_t>, bool> = true>
ALGEBRA_HOST_DEVICE inline array_t<scalar_t, N> normalize(
    const array_t<scalar_t, N> &v) {

  const scalar_t oon =
      static_cast<scalar_t>(1.) / algebra::math::sqrt(dot(v, v));
  return v * oon;
}

}  // namespace algebra::cmath
