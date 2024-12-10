
/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/algebra.hpp"

// System include(s)
#include <type_traits>

namespace detray {

namespace detail {
/// The detray boolean (mask) types (can be SIMD)
/// @{
template <typename A, typename = void>
struct get_bool {};

template <concepts::aos_algebra A>
struct get_bool<A> {
    using boolean = bool;
};

template <concepts::soa_algebra A>
struct get_bool<A> {
    using boolean = typename A::boolean;
};
/// @}

}  // namespace detail

template <typename A>
using dbool = typename detail::get_bool<A>::boolean;

// TODO: Move to algebra plugins
namespace detail {

/// boolean utilities
/// @{
constexpr bool any_of(bool b) {
    return b;
}
constexpr bool all_of(bool b) {
    return b;
}
constexpr bool none_of(bool b) {
    return !b;
}

#if (IS_SOA)
template <typename T>
requires Vc::Traits::is_simd_mask<T>::value inline bool any_of(T &&mask) {
    return Vc::any_of(std::forward<T>(mask));
}

template <typename T>
requires Vc::Traits::is_simd_mask<T>::value inline bool all_of(T &&mask) {
    return Vc::all_of(std::forward<T>(mask));
}

template <typename T>
requires Vc::Traits::is_simd_mask<T>::value inline bool none_of(T &&mask) {
    return Vc::none_of(std::forward<T>(mask));
}
#endif
/// @}

}  // namespace detail

}  // namespace detray
