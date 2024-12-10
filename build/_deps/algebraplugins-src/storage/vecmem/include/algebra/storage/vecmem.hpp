/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/static_array.hpp>

// System include(s).
#include <cstddef>

namespace algebra::vecmem {

/// Array type used in the VecMem storage model
template <typename T, std::size_t N>
using storage_type = ::vecmem::static_array<T, N>;
/// Matrix type used in the VecMem storage model
template <typename T, std::size_t ROWS, std::size_t COLS>
using matrix_type = storage_type<storage_type<T, ROWS>, COLS>;

/// 3-element "vector" type, using @c vecmem::static_array
template <typename T>
using vector3 = storage_type<T, 3>;
/// Point in 3D space, using @c vecmem::static_array
template <typename T>
using point3 = vector3<T>;
/// 2-element "vector" type, using @c vecmem::static_array
template <typename T>
using vector2 = storage_type<T, 2>;
/// Point in 2D space, using @c vecmem::static_array
template <typename T>
using point2 = vector2<T>;

}  // namespace algebra::vecmem
