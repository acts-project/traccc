/** Algebra plugins, part of the ACTS project
 *
 * (c) 2020-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/common.hpp"
#include "algebra/qualifiers.hpp"

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

// System include(s).
#include <type_traits>

namespace algebra::eigen::math {

/** This method retrieves phi from a vector, vector base with rows >= 2
 *
 * @param v the input vector
 **/
template <
    typename derived_type,
    std::enable_if_t<Eigen::MatrixBase<derived_type>::RowsAtCompileTime >= 2,
                     bool> = true>
ALGEBRA_HOST_DEVICE inline auto phi(
    const Eigen::MatrixBase<derived_type> &v) noexcept {

  return algebra::math::atan2(v[1], v[0]);
}

/** This method retrieves theta from a vector, vector base with rows >= 3
 *
 * @param v the input vector
 **/
template <
    typename derived_type,
    std::enable_if_t<Eigen::MatrixBase<derived_type>::RowsAtCompileTime >= 3,
                     bool> = true>
ALGEBRA_HOST_DEVICE inline auto theta(
    const Eigen::MatrixBase<derived_type> &v) noexcept {

  return algebra::math::atan2(algebra::math::sqrt(v[0] * v[0] + v[1] * v[1]),
                              v[2]);
}

/** This method retrieves the perpendicular magnitude of a vector with rows >= 2
 *
 * @param v the input vector
 **/
template <
    typename derived_type,
    std::enable_if_t<Eigen::MatrixBase<derived_type>::RowsAtCompileTime >= 2,
                     bool> = true>
ALGEBRA_HOST_DEVICE inline auto perp(
    const Eigen::MatrixBase<derived_type> &v) noexcept {

  return algebra::math::sqrt(v[0] * v[0] + v[1] * v[1]);
}

/** This method retrieves the norm of a vector, no dimension restriction
 *
 * @param v the input vector
 **/
template <typename derived_type>
ALGEBRA_HOST_DEVICE inline auto norm(const Eigen::MatrixBase<derived_type> &v) {

  return v.norm();
}

/** This method retrieves the pseudo-rapidity from a vector or vector base with
 * rows >= 3
 *
 * @param v the input vector
 **/
template <
    typename derived_type,
    std::enable_if_t<Eigen::MatrixBase<derived_type>::RowsAtCompileTime >= 3,
                     bool> = true>
ALGEBRA_HOST_DEVICE inline auto eta(
    const Eigen::MatrixBase<derived_type> &v) noexcept {

  return algebra::math::atanh(v[2] / v.norm());
}

/// Functor used to access elements of Eigen matrices
struct element_getter {
  /// Get non-const access to a matrix element
  template <typename derived_type, typename size_type_1, typename size_type_2,
            std::enable_if_t<
                std::is_base_of<
                    Eigen::DenseCoeffsBase<derived_type, Eigen::WriteAccessors>,
                    Eigen::MatrixBase<derived_type> >::value &&
                    std::is_convertible<size_type_1, Eigen::Index>::value &&
                    std::is_convertible<size_type_2, Eigen::Index>::value,
                bool> = true>
  ALGEBRA_HOST_DEVICE inline auto &operator()(
      Eigen::MatrixBase<derived_type> &m, size_type_1 row,
      size_type_2 col) const {

    return m(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
  }
  /// Get const access to a matrix element
  template <typename derived_type, typename size_type_1, typename size_type_2,
            std::enable_if_t<
                std::is_convertible<size_type_1, Eigen::Index>::value &&
                    std::is_convertible<size_type_2, Eigen::Index>::value,
                bool> = true>
  ALGEBRA_HOST_DEVICE inline auto operator()(
      const Eigen::MatrixBase<derived_type> &m, size_type_1 row,
      size_type_2 col) const {

    return m(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
  }
};  // struct element_getter

/// Function extracting an element from a matrix (const)
template <
    typename derived_type, typename size_type_1, typename size_type_2,
    std::enable_if_t<std::is_convertible<size_type_1, Eigen::Index>::value &&
                         std::is_convertible<size_type_2, Eigen::Index>::value,
                     bool> = true>
ALGEBRA_HOST_DEVICE inline auto element(
    const Eigen::MatrixBase<derived_type> &m, size_type_1 row,
    size_type_2 col) {

  return element_getter()(m, row, col);
}

/// Function extracting an element from a matrix (non-const)
template <typename derived_type, typename size_type_1, typename size_type_2,
          std::enable_if_t<
              std::is_base_of<
                  Eigen::DenseCoeffsBase<derived_type, Eigen::WriteAccessors>,
                  Eigen::MatrixBase<derived_type> >::value &&
                  std::is_convertible<size_type_1, Eigen::Index>::value &&
                  std::is_convertible<size_type_2, Eigen::Index>::value,
              bool> = true>
ALGEBRA_HOST_DEVICE inline auto &element(Eigen::MatrixBase<derived_type> &m,
                                         size_type_1 row, size_type_2 col) {

  return element_getter()(m, row, col);
}

/// Functor used to extract a block from Eigen matrices
struct block_getter {
  template <int kROWS, int kCOLS, typename matrix_type, typename size_type_1,
            typename size_type_2,
            std::enable_if_t<
                std::is_convertible<size_type_1, Eigen::Index>::value &&
                    std::is_convertible<size_type_2, Eigen::Index>::value,
                bool> = true>
  ALGEBRA_HOST_DEVICE auto operator()(const matrix_type &m, size_type_1 row,
                                      size_type_2 col) const {

    return m.template block<kROWS, kCOLS>(row, col);
  }
};  // struct block_getter

}  // namespace algebra::eigen::math