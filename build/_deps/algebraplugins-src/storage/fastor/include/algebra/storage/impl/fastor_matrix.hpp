/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Fastor include(s).
#ifdef _MSC_VER
#pragma warning(disable : 4244 4701 4702)
#endif  // MSVC
#include <Fastor/Fastor.h>
#ifdef _MSC_VER
#pragma warning(default : 4244 4701 4702)
#endif  // MSVC

// System include(s).
#include <cstddef>

namespace algebra::fastor {

/// Fastor Matrix type
///
/// This class is needed because Fastor differentiates between normal
/// matrix-matrix multiplication and element-wise matrix-matrix multiplication.
///
/// In Fastor, the former is performed by using `operator%` or the
/// `Fastor::matmul()` function, whereas the latter is done using `operator*`.
/// However, the algebra-plugins repository expects operator* for normal
/// matrix-matrix multiplication. To resolve this issue, this wrapper class
/// around `Fastor::Tensor` was created.
///
/// The class inherits from `Fastor::Tensor` because we want objects of this
/// class to behave the way a `Fastor::Tensor` would. Inheriting from
/// `Fastor::Tensor` allows this class to reuse all the functions defined in the
/// parent class (i.e. `Fastor::Tensor`).
template <typename T, std::size_t M1, std::size_t N>
class Matrix : public Fastor::Tensor<T, M1, N> {

 public:
  /// Inherit all constructors from the base class
  using Fastor::Tensor<T, M1, N>::Tensor;

  /// When we encounter an `operator*` function call between Matrix objects, we
  /// will catch it and handle it correctly by invoking `Fastor::matmul()`.
  ///
  /// The data type contained in the `other` has a separate template parameter
  /// dedicated to it because in certain cases, we might want to multiply say, a
  /// float matrix with a double matrix and not have it produce a compilation
  /// error.
  ///
  /// The `static_cast` is there to signal both to the compiler and the reader
  /// that we wish to interpret the `Matrix` object as a `Fastor::Tensor` here.
  template <typename U, std::size_t M2,
            std::enable_if_t<std::is_convertible_v<U, T>, bool> = true>
  inline Matrix<T, M1, M2> operator*(const Matrix<U, N, M2>& other) const {
    return Fastor::matmul(static_cast<Fastor::Tensor<T, M1, N>>(*this),
                          static_cast<Fastor::Tensor<T, N, M2>>(other));
  }

  /// When we encounter an `operator*` function call between a Matrix object and
  /// a `Fastor::Tensor` with one dimensional parameter (i.e. a vector), we will
  /// catch it and handle it correctly by invoking `Fastor::matmul()`.
  ///
  /// The data type contained in the `other` has a separate template parameter
  /// dedicated to it because in certain cases, we might want to multiply say, a
  /// float matrix with a double vector and not have it produce a compilation
  /// error.
  ///
  /// The `static_cast` is there to signal both to the compiler and the reader
  /// that we wish to interpret the `Matrix` object as a `Fastor::Tensor` here.
  template <typename U,
            std::enable_if_t<std::is_convertible_v<U, T>, bool> = true>
  inline Fastor::Tensor<T, M1> operator*(
      const Fastor::Tensor<U, N>& other) const {
    return Fastor::matmul(static_cast<Fastor::Tensor<T, M1, N>>(*this),
                          static_cast<Fastor::Tensor<T, N>>(other));
  }

};  // class Matrix

}  // namespace algebra::fastor
