/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/qualifiers.hpp"

// System include(s)
#include <type_traits>

namespace algebra::cmath::matrix::determinant {

/// "Determinant getter", assuming a N X N matrix
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

  template <size_type N>
  ALGEBRA_HOST_DEVICE inline scalar_t operator()(
      const matrix_type<N, N> &m) const {
    return determinant_getter_helper<N>()(m);
  }

  template <size_type N, typename Enable = void>
  struct determinant_getter_helper;

  template <size_type N>
  struct determinant_getter_helper<N, typename std::enable_if_t<N == 1>> {
    template <class input_matrix_type>
    ALGEBRA_HOST_DEVICE inline scalar_t operator()(
        const input_matrix_type &m) const {
      return element_getter()(m, 0, 0);
    }
  };

  template <size_type N>
  struct determinant_getter_helper<N, typename std::enable_if_t<N != 1>> {

    template <class input_matrix_type>
    ALGEBRA_HOST_DEVICE inline scalar_t operator()(
        const input_matrix_type &m) const {

      scalar_t D = 0;

      // To store cofactors
      matrix_type<N, N> temp;

      // To store sign multiplier
      int sign = 1;

      // Iterate for each element of first row
      for (size_type col = 0; col < N; col++) {
        // Getting Cofactor of A[0][f]
        this->get_cofactor(m, temp, size_type(0), col);
        D += sign * element_getter()(m, 0, col) *
             determinant_getter_helper<N - 1>()(temp);

        // terms are to be added with alternate sign
        sign = -sign;
      }

      return D;
    }

    template <class input_matrix_type>
    ALGEBRA_HOST_DEVICE inline void get_cofactor(const input_matrix_type &m,
                                                 matrix_type<N, N> &temp,
                                                 size_type p,
                                                 size_type q) const {

      size_type i = 0, j = 0;

      // Looping for each element of the matrix
      for (size_type row = 0; row < N; row++) {
        for (size_type col = 0; col < N; col++) {
          //  Copying into temporary matrix only those element
          //  which are not in given row and column
          if (row != p && col != q) {
            element_getter()(temp, i, j++) = element_getter()(m, row, col);

            // Row is filled, so increase row index and
            // reset col index
            if (j == N - 1) {
              j = 0;
              i++;
            }
          }
        }
      }
    }
  };
};

}  // namespace algebra::cmath::matrix::determinant
