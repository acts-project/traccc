/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/storage/impl/vc_array4.hpp"

// System include(s).
#include <array>
#include <cstddef>

// Vc include(s).
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif  // MSVC
#include <Vc/Vc>
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // MSVC

namespace algebra::vc {

/// size type for Vc storage model
using size_type = std::size_t;
/// Array type used in the Vc storage model
template <typename T, size_type N>
using storage_type = Vc::SimdArray<T, N>;
/// Matrix type used in the Vc storage model
template <typename T, size_type ROWS, size_type COLS>
using matrix_type = Vc::array<Vc::array<T, ROWS>, COLS>;

/// 3-element "vector" type, using @c algebra::vc::array4
template <typename T>
using vector3 = Vc::array<T, 3>;
/// Point in 3D space, using @c algebra::vc::array4
template <typename T>
using point3 = vector3<T>;
/// 2-element "vector" type, using @c std::array
template <typename T>
using vector2 = Vc::array<T, 2>;
/// Point in 2D space, using @c std::array
template <typename T>
using point2 = vector2<T>;

}  // namespace algebra::vc