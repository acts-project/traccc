/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/algorithms/matrix/determinant/cofactor.hpp"
#include "algebra/qualifiers.hpp"

// System include(s)
#include <type_traits>

namespace algebra::cmath::matrix {

namespace adjoint {

/// "Adjoint getter", assuming a N X N matrix
template <typename size_type,
          template <typename, size_type, size_type> class matrix_t,
          typename scalar_t, class element_getter_t>
struct cofactor {

  /// Function (object) used for accessing a matrix element
  using element_getter = element_getter_t;

  /// 2D matrix type
  template <size_type ROWS, size_type COLS>
  using matrix_type = matrix_t<scalar_t, ROWS, COLS>;

  template <size_type N>
  ALGEBRA_HOST_DEVICE inline matrix_type<N, N> operator()(
      const matrix_type<N, N> &m) const {
    return adjoint_getter_helper<N>()(m);
  }

  template <size_type N, typename Enable = void>
  struct adjoint_getter_helper;

  template <size_type N>
  struct adjoint_getter_helper<N, typename std::enable_if_t<N == 1>> {

    ALGEBRA_HOST_DEVICE inline matrix_type<N, N> operator()(
        const matrix_type<N, N> & /*m*/) const {
      matrix_type<N, N> ret;
      element_getter()(ret, 0, 0) = 1;
      return ret;
    }
  };

  template <size_type N>
  struct adjoint_getter_helper<N, typename std::enable_if_t<N != 1>> {

    using determinant_getter =
        determinant::cofactor<size_type, matrix_t, scalar_t, element_getter_t>;

    ALGEBRA_HOST_DEVICE inline matrix_type<N, N> operator()(
        const matrix_type<N, N> &m) const {

      matrix_type<N, N> adj;

      // temp is used to store cofactors of m
      int sign = 1;

      // To store cofactors
      matrix_type<N, N> temp;

      for (size_type i = 0; i < N; i++) {
        for (size_type j = 0; j < N; j++) {
          // Get cofactor of m[i][j]
          typename determinant_getter::template determinant_getter_helper<N>()
              .get_cofactor(m, temp, i, j);

          // sign of adj[j][i] positive if sum of row
          // and column indexes is even.
          sign = ((i + j) % 2 == 0) ? 1 : -1;

          // Interchanging rows and columns to get the
          // transpose of the cofactor matrix
          element_getter()(adj, j, i) =
              sign *
              typename determinant_getter::template determinant_getter_helper<
                  N - 1>()(temp);
        }
      }

      return adj;
    }
  };
};

}  // namespace adjoint

namespace inverse {

/// "inverse getter", assuming a N X N matrix
template <typename size_type,
          template <typename, size_type, size_type> class matrix_t,
          typename scalar_t, class element_getter_t, size_type... Ds>
struct cofactor {

  using _dims = std::integer_sequence<size_type, Ds...>;

  /// Function (object) used for accessing a matrix element
  using element_getter = element_getter_t;

  /// 2D matrix type
  template <size_type ROWS, size_type COLS>
  using matrix_type = matrix_t<scalar_t, ROWS, COLS>;

  using determinant_getter =
      determinant::cofactor<size_type, matrix_t, scalar_t, element_getter_t>;

  using adjoint_getter =
      adjoint::cofactor<size_type, matrix_t, scalar_t, element_getter_t>;

  template <size_type N>
  ALGEBRA_HOST_DEVICE inline matrix_type<N, N> operator()(
      const matrix_type<N, N> &m) const {

    matrix_type<N, N> ret;

    // Find determinant of A
    scalar_t det = determinant_getter()(m);

    // TODO: handle singular matrix error
    // if (det == 0) {
    // return ret;
    //}

    auto adj = adjoint_getter()(m);

    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
    for (size_type i = 0; i < N; i++) {
      for (size_type j = 0; j < N; j++) {
        element_getter()(ret, j, i) = element_getter()(adj, j, i) / det;
      }
    }

    return ret;
  }
};

}  // namespace inverse

}  // namespace algebra::cmath::matrix
