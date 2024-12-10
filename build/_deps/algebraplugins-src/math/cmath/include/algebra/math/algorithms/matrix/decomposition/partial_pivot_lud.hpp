/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/common.hpp"
#include "algebra/qualifiers.hpp"

// System include(s).
#include <array>

namespace algebra::cmath::matrix::decomposition {

/// "Partial Pivot LU Decomposition", assuming a N X N matrix
template <typename size_type,
          template <typename, size_type, size_type> class matrix_t,
          typename scalar_t, class element_getter_t, size_type... Ds>
struct partial_pivot_lud {

  using _dims = std::integer_sequence<size_type, Ds...>;

  /// Function (object) used for accessing a matrix element
  using element_getter = element_getter_t;

  /// 2D matrix type
  template <size_type ROWS, size_type COLS>
  using matrix_type = matrix_t<scalar_t, ROWS, COLS>;

  template <size_type N>
  struct lud {
    // LU decomposition matrix, equal to (L - I) + U, where the diagonal
    // components of L is always 1
    matrix_type<N, N> lu;

    // Permutation vector
    matrix_type<1, N> P;

    // Number of pivots
    int n_pivot = 0;
  };

  template <size_type N>
  ALGEBRA_HOST_DEVICE inline lud<N> operator()(
      const matrix_type<N, N>& m) const {
    // LU decomposition matrix
    matrix_type<N, N> lu = m;

    // Permutation
    matrix_type<1, N> P;

    // Max index and value
    size_type max_idx;
    scalar_t max_val;
    scalar_t abs_val;

    // Number of pivoting
    int n_pivot = N;

    // Rows for swapping
    matrix_type<1, N> row_0;
    matrix_type<1, N> row_1;

    // Unit permutation matrix, P[N] initialized with N
    for (size_type i = 0; i < N; i++) {
      element_getter_t()(P, 0, i) = static_cast<scalar_t>(i);
    }

    for (size_type i = 0; i < N; i++) {
      max_val = 0;
      max_idx = i;

      for (size_type k = i; k < N; k++) {
        abs_val = math::fabs(element_getter()(lu, k, i));

        if (abs_val > max_val) {

          max_val = abs_val;
          max_idx = k;
        }
      }

      if (max_idx != i) {
        // Pivoting P
        auto j = element_getter_t()(P, 0, i);

        element_getter_t()(P, 0, i) = element_getter_t()(P, 0, max_idx);
        element_getter_t()(P, 0, max_idx) = j;

        // Pivoting rows of A
        for (size_type q = 0; q < N; q++) {
          element_getter_t()(row_0, 0, q) = element_getter_t()(lu, i, q);
          element_getter_t()(row_1, 0, q) = element_getter_t()(lu, max_idx, q);
        }
        for (size_type q = 0; q < N; q++) {
          element_getter_t()(lu, i, q) = element_getter_t()(row_1, 0, q);
          element_getter_t()(lu, max_idx, q) = element_getter_t()(row_0, 0, q);
        }

        // counting pivots starting from N (for determinant)
        n_pivot++;
      }

      for (size_type j = i + 1; j < N; j++) {
        // m[j][i] /= m[i][i];
        element_getter_t()(lu, j, i) /= element_getter_t()(lu, i, i);

        for (size_type k = i + 1; k < N; k++) {
          // m[j][k] -= m[j][i] * m[i][k];
          element_getter_t()(lu, j, k) -=
              element_getter_t()(lu, j, i) * element_getter_t()(lu, i, k);
        }
      }
    }

    return {lu, P, n_pivot};
  }
};

}  // namespace algebra::cmath::matrix::decomposition