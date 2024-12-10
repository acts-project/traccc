/** Algebra plugins, part of the ACTS project
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/common.hpp"
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
#include <cstddef>  // for the std::size_t type
#include <type_traits>

namespace algebra::fastor::math {

/** This method retrieves phi from a vector, vector base with rows >= 2
 *
 * @param v the input vector
 **/
template <typename scalar_t, auto N, std::enable_if_t<N >= 2, bool> = true>
ALGEBRA_HOST inline scalar_t phi(
    const Fastor::Tensor<scalar_t, N> &v) noexcept {

  return algebra::math::atan2(v[1], v[0]);
}

/** This method retrieves theta from a vector, vector base with rows >= 3
 *
 * @param v the input vector
 **/
template <typename scalar_t, auto N, std::enable_if_t<N >= 3, bool> = true>
ALGEBRA_HOST inline scalar_t theta(
    const Fastor::Tensor<scalar_t, N> &v) noexcept {

  return algebra::math::atan2(Fastor::norm(v(Fastor::fseq<0, 2>())), v[2]);
}

/** This method retrieves the perpenticular magnitude of a vector with rows >= 2
 *
 * @param v the input vector
 **/
template <typename scalar_t, auto N, std::enable_if_t<N >= 2, bool> = true>
ALGEBRA_HOST inline scalar_t perp(
    const Fastor::Tensor<scalar_t, N> &v) noexcept {

  return algebra::math::sqrt(
      Fastor::inner(v(Fastor::fseq<0, 2>()), v(Fastor::fseq<0, 2>())));
}

/** This method retrieves the norm of a vector, no dimension restriction
 *
 * @param v the input vector
 **/
template <typename scalar_t, auto N>
ALGEBRA_HOST inline scalar_t norm(const Fastor::Tensor<scalar_t, N> &v) {

  return Fastor::norm(v);
}

/** This method retrieves the pseudo-rapidity from a vector or vector base with
 *rows >= 3
 *
 * @param v the input vector
 **/
template <typename scalar_t, auto N, std::enable_if_t<N >= 3, bool> = true>
ALGEBRA_HOST inline scalar_t eta(
    const Fastor::Tensor<scalar_t, N> &v) noexcept {

  return algebra::math::atanh(v[2] / Fastor::norm(v));
}

/// Functor used to access elements of Fastor matrices
template <typename scalar_t>
struct element_getter {

  template <std::size_t ROWS, std::size_t COLS>
  using matrix_type = Fastor::Tensor<scalar_t, ROWS, COLS>;

  template <std::size_t ROWS, std::size_t COLS>
  ALGEBRA_HOST_DEVICE inline scalar_t &operator()(matrix_type<ROWS, COLS> &m,
                                                  std::size_t row,
                                                  std::size_t col) const {

    assert(row < ROWS);
    assert(col < COLS);
    return m(row, col);
  }

  template <std::size_t ROWS, std::size_t COLS>
  ALGEBRA_HOST_DEVICE inline scalar_t operator()(
      const matrix_type<ROWS, COLS> &m, std::size_t row,
      std::size_t col) const {

    assert(row < ROWS);
    assert(col < COLS);
    return m(row, col);
  }
};  // element_getter

/// Function extracting an element from a matrix (const)
template <typename scalar_t, std::size_t ROWS, std::size_t COLS>
ALGEBRA_HOST_DEVICE inline scalar_t element(
    const Fastor::Tensor<scalar_t, ROWS, COLS> &m, std::size_t row,
    std::size_t col) {

  return element_getter<scalar_t>()(m, row, col);
}

/// Function extracting an element from a matrix (non-const)
template <typename scalar_t, std::size_t ROWS, std::size_t COLS>
ALGEBRA_HOST_DEVICE inline scalar_t &element(
    Fastor::Tensor<scalar_t, ROWS, COLS> &m, std::size_t row, std::size_t col) {

  return element_getter<scalar_t>()(m, row, col);
}

/// Functor used to extract a block from Fastor matrices
template <typename scalar_t>
struct block_getter {

  template <std::size_t ROWS, std::size_t COLS>
  using matrix_type = Fastor::Tensor<scalar_t, ROWS, COLS>;

  template <std::size_t ROWS, std::size_t COLS, class input_matrix_type>
  ALGEBRA_HOST_DEVICE matrix_type<ROWS, COLS> operator()(
      const input_matrix_type &m, std::size_t row, std::size_t col) const {

    return m(Fastor::seq(row, row + ROWS), Fastor::seq(col, col + COLS));
  }
};  // struct block_getter

}  // namespace algebra::fastor::math
