/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/vc_soa.hpp"
#include "algebra/storage/vc_soa.hpp"

// System include(s).
#include <cassert>
#include <type_traits>

/// @name Operators on @c algebra::storage::vector types
/// @{

using algebra::storage::operator*;
using algebra::storage::operator/;
using algebra::storage::operator-;
using algebra::storage::operator+;

/// @}

namespace algebra {
namespace vc_soa {

/// @name Vc based transforms on @c algebra::vc_soa types
/// @{

template <typename T>
using transform3 = math::transform3<T>;

/// @}

}  // namespace vc_soa

namespace getter {

/// @name Getter functions on @c algebra::vc_soa types
/// @{

using vc_soa::math::eta;
using vc_soa::math::norm;
using vc_soa::math::perp;
using vc_soa::math::phi;
using vc_soa::math::theta;

/// @}

}  // namespace getter

namespace vector {

/// @name Vector functions on @c algebra::vc_soa types
/// @{

using vc_soa::math::cross;
using vc_soa::math::dot;
using vc_soa::math::normalize;

/// @}

}  // namespace vector

namespace matrix {

using size_type = vc_soa::size_type;

template <typename T, size_type N>
using array_type = vc_soa::storage_type<T, N>;

template <typename T, size_type ROWS, size_type COLS>
using matrix_type = vc_soa::matrix_type<T, ROWS, COLS>;

}  // namespace matrix

}  // namespace algebra
