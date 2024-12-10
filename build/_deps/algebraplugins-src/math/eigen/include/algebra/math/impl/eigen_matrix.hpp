/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/qualifiers.hpp"
#include "algebra/storage/eigen.hpp"

// Eigen include(s).
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif  // MSVC
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20012
#endif  // __NVCC_DIAG_PRAGMA_SUPPORT__
#include <Eigen/Core>
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // MSVC
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diagnostic pop
#endif  // __NVCC_DIAG_PRAGMA_SUPPORT__

namespace algebra::eigen::matrix {

/// "Matrix actor", assuming an Eigen matrix
template <typename scalar_t>
struct actor {

  /// Size type
  using size_ty = int;

  /// Scalar type
  using scalar_type = scalar_t;

  /// 2D matrix type
  template <int ROWS, int COLS>
  using matrix_type = algebra::eigen::matrix_type<scalar_t, ROWS, COLS>;

  /// Array type
  template <size_ty N>
  using array_type = storage_type<scalar_type, N>;

  /// 3-element "vector" type
  using vector3 = array_type<3>;

  /// Operator getting a reference to one element of a non-const matrix
  template <int ROWS, int COLS, typename size_type_1, typename size_type_2,
            std::enable_if_t<
                std::is_convertible<size_type_1, Eigen::Index>::value &&
                    std::is_convertible<size_type_2, Eigen::Index>::value,
                bool> = true>
  ALGEBRA_HOST_DEVICE inline scalar_t &element(matrix_type<ROWS, COLS> &m,
                                               size_type_1 row,
                                               size_type_2 col) const {
    return m(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
  }

  /// Operator getting one value of a const matrix
  template <int ROWS, int COLS, typename size_type_1, typename size_type_2,
            std::enable_if_t<
                std::is_convertible<size_type_1, Eigen::Index>::value &&
                    std::is_convertible<size_type_2, Eigen::Index>::value,
                bool> = true>
  ALGEBRA_HOST_DEVICE inline scalar_t element(const matrix_type<ROWS, COLS> &m,
                                              size_type_1 row,
                                              size_type_2 col) const {
    return m(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
  }

  /// Operator getting a block of a const matrix
  template <int ROWS, int COLS, class input_matrix_type, typename size_type_1,
            typename size_type_2,
            std::enable_if_t<
                std::is_convertible<size_type_1, Eigen::Index>::value &&
                    std::is_convertible<size_type_2, Eigen::Index>::value,
                bool> = true>
  ALGEBRA_HOST_DEVICE matrix_type<ROWS, COLS> block(const input_matrix_type &m,
                                                    size_type_1 row,
                                                    size_type_2 col) const {
    return m.template block<ROWS, COLS>(static_cast<Eigen::Index>(row),
                                        static_cast<Eigen::Index>(col));
  }

  /// Operator setting a block
  template <int ROWS, int COLS, class input_matrix_type, typename size_type_1,
            typename size_type_2,
            std::enable_if_t<
                std::is_convertible<size_type_1, Eigen::Index>::value &&
                    std::is_convertible<size_type_2, Eigen::Index>::value,
                bool> = true>
  ALGEBRA_HOST_DEVICE void set_block(input_matrix_type &m,
                                     const matrix_type<ROWS, COLS> &b,
                                     size_type_1 row, size_type_2 col) const {
    m.template block<ROWS, COLS>(static_cast<Eigen::Index>(row),
                                 static_cast<Eigen::Index>(col)) = b;
  }

  // Create zero matrix
  template <int ROWS, int COLS>
  ALGEBRA_HOST_DEVICE inline matrix_type<ROWS, COLS> zero() const {
    return matrix_type<ROWS, COLS>::Zero();
  }

  // Create identity matrix
  template <int ROWS, int COLS>
  ALGEBRA_HOST_DEVICE inline matrix_type<ROWS, COLS> identity() const {
    return matrix_type<ROWS, COLS>::Identity();
  }

  // Set input matrix as zero matrix
  template <int ROWS, int COLS>
  ALGEBRA_HOST_DEVICE inline void set_zero(matrix_type<ROWS, COLS> &m) const {
    m.setZero();
  }

  // Set input matrix as identity matrix
  template <int ROWS, int COLS>
  ALGEBRA_HOST_DEVICE inline void set_identity(
      matrix_type<ROWS, COLS> &m) const {
    m.setIdentity();
  }

  // Create transpose matrix
  template <int ROWS, int COLS>
  ALGEBRA_HOST_DEVICE inline matrix_type<COLS, ROWS> transpose(
      const matrix_type<ROWS, COLS> &m) const {
    return m.transpose();
  }

  // Get determinant
  template <int N>
  ALGEBRA_HOST_DEVICE inline scalar_t determinant(
      const matrix_type<N, N> &m) const {
    return m.determinant();
  }

  // Create inverse matrix
  template <int N>
  ALGEBRA_HOST_DEVICE inline matrix_type<N, N> inverse(
      const matrix_type<N, N> &m) const {
    return m.inverse();
  }
};

}  // namespace algebra::eigen::matrix
