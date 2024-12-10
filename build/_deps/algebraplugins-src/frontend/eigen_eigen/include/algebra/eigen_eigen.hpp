/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/cmath.hpp"
#include "algebra/math/eigen.hpp"
#include "algebra/storage/eigen.hpp"

// Eigen include(s).
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif  // MSVC
#include <Eigen/Core>
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // MSVC

namespace algebra {

namespace getter {

/// @name Getter functions on @c algebra::eigen::storage_type
/// @{

using eigen::math::eta;
using eigen::math::norm;
using eigen::math::perp;
using eigen::math::phi;
using eigen::math::theta;

/// @}

/// Function extracting a slice from the matrix used by
/// @c algebra::eigen::transform3
template <unsigned int SIZE, typename derived_type>
ALGEBRA_HOST_DEVICE inline auto vector(const Eigen::MatrixBase<derived_type>& m,
                                       std::size_t row, std::size_t col) {

  return m.template block<SIZE, 1>(static_cast<Eigen::Index>(row),
                                   static_cast<Eigen::Index>(col));
}

/// @name Getter functions on @c algebra::eigen::matrix_type
/// @{

using eigen::math::element;

/// @}

}  // namespace getter

namespace vector {

/// @name Vector functions on @c algebra::eigen::storage_type
/// @{

using eigen::math::cross;
using eigen::math::dot;
using eigen::math::normalize;

/// @}

}  // namespace vector

namespace matrix {

template <typename scalar_t>
using actor = eigen::matrix::actor<scalar_t>;

}  // namespace matrix

namespace eigen {

template <typename T>
using transform3 = math::transform3<T, matrix::actor<T>>;

}  // namespace eigen

}  // namespace algebra
