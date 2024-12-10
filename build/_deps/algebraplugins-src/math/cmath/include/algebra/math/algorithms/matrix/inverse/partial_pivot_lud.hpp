/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/algorithms/matrix/decomposition/partial_pivot_lud.hpp"
#include "algebra/qualifiers.hpp"

namespace algebra::cmath::matrix::inverse {

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

  using decomposition_t =
      typename algebra::cmath::matrix::decomposition::partial_pivot_lud<
          size_type, matrix_t, scalar_t, element_getter_t>;

  template <size_type N>
  ALGEBRA_HOST_DEVICE inline matrix_type<N, N> operator()(
      const matrix_type<N, N>& m) const {
    const typename decomposition_t::template lud<N> decomp_res =
        decomposition_t()(m);

    // Get the LU decomposition matrix equal to (L - I) + U
    const auto& lu = decomp_res.lu;

    // Permutation vector
    const auto& P = decomp_res.P;

    // Inverse matrix
    matrix_type<N, N> inv;

    // Calculate inv(A) = inv(U) * inv(L) * P;
    for (size_type j = 0; j < N; j++) {
      for (size_type i = 0; i < N; i++) {
        element_getter_t()(inv, i, j) =
            static_cast<size_type>(element_getter_t()(P, 0, i)) == j
                ? static_cast<scalar_t>(1.0)
                : static_cast<scalar_t>(0.0);

        for (size_type k = 0; k < i; k++) {
          element_getter_t()(inv, i, j) -=
              element_getter_t()(lu, i, k) * element_getter_t()(inv, k, j);
        }
      }

      for (size_type i = N - 1; int(i) >= 0; i--) {
        for (size_type k = i + 1; k < N; k++) {
          element_getter_t()(inv, i, j) -=
              element_getter_t()(lu, i, k) * element_getter_t()(inv, k, j);
        }
        element_getter_t()(inv, i, j) /= element_getter_t()(lu, i, i);
      }
    }

    return inv;
  }
};

}  // namespace algebra::cmath::matrix::inverse