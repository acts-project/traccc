/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// ROOT/Smatrix include(s).
#include <Math/SMatrix.h>
#include <Math/SVector.h>

// System include(s).
#include <cstddef>

namespace algebra::smatrix {

/// size type for SMatrix storage model
using size_type = unsigned int;
/// Array type used in the SMatrix storage model
template <typename T, size_type N>
using storage_type = ROOT::Math::SVector<T, N>;
/// Matrix type used in the SMatrix storage model
template <typename T, size_type ROWS, size_type COLS>
using matrix_type = ROOT::Math::SMatrix<T, ROWS, COLS>;

/// 3-element "vector" type, using @c ROOT::Math::SVector
template <typename T>
using vector3 = storage_type<T, 3>;
/// Point in 3D space, using @c ROOT::Math::SVector
template <typename T>
using point3 = vector3<T>;
/// 2-element "vector" type, using @c ROOT::Math::SVector
template <typename T>
using vector2 = storage_type<T, 2>;
/// Point in 2D space, using @c ROOT::Math::SVector
template <typename T>
using point2 = vector2<T>;

}  // namespace algebra::smatrix
