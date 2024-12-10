/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/storage/impl/eigen_array.hpp"

// System include(s).
#include <cstddef>

namespace algebra::eigen {

/// size type for Eigen storage model
using size_type = int;
/// Array type used in the Eigen storage model
template <typename T, size_type N>
using storage_type = array<T, N>;
/// Matrix type used in the Eigen storage model
/// If the number of rows is 1, make it RowMajor
template <typename T, size_type ROWS, size_type COLS>
using matrix_type = Eigen::Matrix<T, ROWS, COLS, (ROWS == 1), ROWS, COLS>;

/// 3-element "vector" type, using @c algebra::eigen::array
template <typename T>
using vector3 = storage_type<T, 3>;
/// Point in 3D space, using @c algebra::eigen::array
template <typename T>
using point3 = vector3<T>;
/// 2-element "vector" type, using @c algebra::eigen::array
template <typename T>
using vector2 = storage_type<T, 2>;
/// Point in 2D space, using @c algebra::eigen::array
template <typename T>
using point2 = vector2<T>;

}  // namespace algebra::eigen
