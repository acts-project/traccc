/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
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

namespace algebra::eigen::math {

/** Get a normalized version of the input vector
 *
 * @tparam derived_type is the matrix template
 *
 * @param v the input vector
 **/
template <typename derived_type>
ALGEBRA_HOST_DEVICE inline auto normalize(
    const Eigen::MatrixBase<derived_type> &v) {
  return v.normalized();
}

/** Dot product between two input vectors
 *
 * @tparam derived_type_lhs is the first matrix (expression) template
 * @tparam derived_type_rhs is the second matrix (expression) template
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return the scalar dot product value
 **/
template <typename derived_type_lhs, typename derived_type_rhs>
ALGEBRA_HOST_DEVICE inline auto dot(
    const Eigen::MatrixBase<derived_type_lhs> &a,
    const Eigen::MatrixBase<derived_type_rhs> &b) {
  return a.dot(b);
}

/** Cross product between two input vectors
 *
 * @tparam derived_type_lhs is the first matrix (expression) template
 * @tparam derived_type_rhs is the second matrix (expression) template
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return a vector (expression) representing the cross product
 **/
template <typename derived_type_lhs, typename derived_type_rhs>
ALGEBRA_HOST_DEVICE inline auto cross(
    const Eigen::MatrixBase<derived_type_lhs> &a,
    const Eigen::MatrixBase<derived_type_rhs> &b) {
  return a.cross(b);
}

}  // namespace algebra::eigen::math
