/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/storage/impl/fastor_matrix.hpp"

// System include(s).
#include <cstddef>

namespace algebra::fastor {

/// size type for Fastor storage model
using size_type = std::size_t;
/// Array type used in the Fastor storage model
template <typename T, size_type N>
using storage_type = Fastor::Tensor<T, N>;
/// Matrix type used in the Fastor storage model
template <typename T, size_type ROWS, size_type COLS>
using matrix_type = Matrix<T, ROWS, COLS>;

/// 3-element "vector" type, using @c Fastor::Tensor
template <typename T>
using vector3 = storage_type<T, 3>;
/// Point in 3D space, using @c Fastor::Tensor
template <typename T>
using point3 = vector3<T>;
/// 2-element "vector" type, using @c Fastor::Tensor
template <typename T>
using vector2 = storage_type<T, 2>;
/// Point in 2D space, using @c Fastor::Tensor
template <typename T>
using point2 = vector2<T>;

}  // namespace algebra::fastor
