/** Algebra plugins, part of the ACTS project
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/qualifiers.hpp"
#include "algebra/storage/fastor.hpp"

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

namespace algebra::fastor::matrix {

/// "Matrix actor", assuming a Fastor matrix
template <typename scalar_t>
struct actor {

  /// Size type
  using size_ty = std::size_t;

  /// Scalar type
  using scalar_type = scalar_t;

  /// 2D matrix type
  template <size_ty ROWS, size_ty COLS>
  using matrix_type = algebra::fastor::matrix_type<scalar_t, ROWS, COLS>;

  /// Vector type
  template <size_ty ROWS>
  using vector_type = Fastor::Tensor<scalar_t, ROWS>;

  /// Array type
  template <size_ty N>
  using array_type = storage_type<scalar_type, N>;

  /// 3-element "vector" type
  using vector3 = array_type<3>;

  /// Operator getting a reference to one element of a non-const matrix
  template <size_ty ROWS, size_ty COLS>
  ALGEBRA_HOST_DEVICE inline scalar_t &element(matrix_type<ROWS, COLS> &m,
                                               size_ty row, size_ty col) const {
    return m(row, col);
  }

  /// Operator getting one value of a const matrix
  template <size_ty ROWS, size_ty COLS>
  ALGEBRA_HOST_DEVICE inline scalar_t element(const matrix_type<ROWS, COLS> &m,
                                              size_ty row, size_ty col) const {
    return m(row, col);
  }

  /// Operator getting a block of a const matrix
  template <size_ty ROWS, size_ty COLS, class input_matrix_type>
  ALGEBRA_HOST_DEVICE matrix_type<ROWS, COLS> block(const input_matrix_type &m,
                                                    size_ty row,
                                                    size_ty col) const {
    // In `Fastor::seq`, the last element is not included.
    // `Fastor::seq` takes `int`s as input, but `row`, `col`, `ROWS`, and `COLS`
    // have type `size_ty`, which is `std::size_t`.
    return m(Fastor::seq(static_cast<int>(row), static_cast<int>(row + ROWS)),
             Fastor::seq(static_cast<int>(col), static_cast<int>(col + COLS)));
  }

  /// Operator setting a block with a matrix
  template <size_ty ROWS, size_ty COLS, class input_matrix_type>
  ALGEBRA_HOST_DEVICE void set_block(input_matrix_type &m,
                                     const matrix_type<ROWS, COLS> &b,
                                     size_ty row, size_ty col) const {
    // In `Fastor::seq`, the last element is not included.
    // `Fastor::seq` takes `int`s as input, but `ROWS` and `COLS` have type
    // `size_ty`, which is `std::size_t`.
    m(Fastor::seq(static_cast<int>(row), static_cast<int>(row + ROWS)),
      Fastor::seq(static_cast<int>(col), static_cast<int>(col + COLS))) = b;
  }

  /// Operator setting a block with a vector
  template <size_ty ROWS, class input_matrix_type>
  ALGEBRA_HOST_DEVICE void set_block(input_matrix_type &m,
                                     const vector_type<ROWS> &b, size_ty row,
                                     size_ty col) const {
    // In `Fastor::seq`, the last element is not included.
    // `Fastor::seq` takes `int`s as input, but `ROWS` and `COLS` have type
    // `size_ty`, which is `std::size_t`.
    m(Fastor::seq(static_cast<int>(row), static_cast<int>(row + ROWS)),
      static_cast<int>(col)) = b;
  }

  // Create zero matrix
  template <size_ty ROWS, size_ty COLS>
  ALGEBRA_HOST_DEVICE inline matrix_type<ROWS, COLS> zero() const {
    return matrix_type<ROWS, COLS>(0);
  }

  // Create identity matrix
  template <size_ty ROWS, size_ty COLS>
  ALGEBRA_HOST_DEVICE inline matrix_type<ROWS, COLS> identity() const {
    // There are 2 identity tensor methods in Fastor, eye() and eye2(). The
    // former is for arbitrary order tensors, whereas the latter is specifically
    // for second order tensors. As such, I chose to use eye2() here because it
    // does less and hence would be faster.

    // eye2() only works for square matrices. The idea is to take the largest
    // dimension of the matrix, make an identity matrix of that dimension, and
    // then return the appropriately sized submatrix of it.
    if constexpr (ROWS >= COLS) {
      matrix_type<ROWS, ROWS> identity_matrix;
      identity_matrix.eye2();
      return matrix_type<ROWS, COLS>(
          identity_matrix(Fastor::fseq<0, ROWS>(), Fastor::fseq<0, COLS>()));
    } else {
      matrix_type<COLS, COLS> identity_matrix;
      identity_matrix.eye2();
      return matrix_type<ROWS, COLS>(
          identity_matrix(Fastor::fseq<0, ROWS>(), Fastor::fseq<0, COLS>()));
    }
  }

  // Set input matrix as zero matrix
  template <size_ty ROWS, size_ty COLS>
  ALGEBRA_HOST_DEVICE inline void set_zero(matrix_type<ROWS, COLS> &m) const {
    m.zeros();
  }

  // Set input matrix as identity matrix
  template <size_ty ROWS, size_ty COLS>
  ALGEBRA_HOST_DEVICE inline void set_identity(
      matrix_type<ROWS, COLS> &m) const {

    m = identity<ROWS, COLS>();
  }

  // Create transpose matrix
  template <size_ty ROWS, size_ty COLS>
  ALGEBRA_HOST_DEVICE inline matrix_type<COLS, ROWS> transpose(
      const matrix_type<ROWS, COLS> &m) const {
    return Fastor::transpose(m);
  }

  // Get determinant
  template <size_ty N>
  ALGEBRA_HOST_DEVICE inline scalar_t determinant(
      const matrix_type<N, N> &m) const {
    return Fastor::determinant(m);
  }

  // Create inverse matrix
  template <size_ty N>
  ALGEBRA_HOST_DEVICE inline matrix_type<N, N> inverse(
      const matrix_type<N, N> &m) const {
    return Fastor::inverse(m);
  }
};

}  // namespace algebra::fastor::matrix
