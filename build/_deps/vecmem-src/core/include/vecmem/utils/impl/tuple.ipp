/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cstddef>
#include <type_traits>

namespace vecmem {
namespace details {

/// Struct used to implement @c vecmem::get in a C++14 style
///
/// @tparam I The index of the tuple element to get
///
template <std::size_t I>
struct tuple_get_impl {

    /// Get the I-th (const) tuple element recursively
    template <typename... Ts>
    VECMEM_HOST_AND_DEVICE static constexpr const auto &get(
        const tuple<Ts...> &t) {
        return tuple_get_impl<I - 1>::get(t.m_tail);
    }
    /// Get the I-th (non-const) tuple element recursively
    template <typename... Ts>
    VECMEM_HOST_AND_DEVICE static constexpr auto &get(tuple<Ts...> &t) {
        return tuple_get_impl<I - 1>::get(t.m_tail);
    }

};  // struct tuple_get_impl

/// Specialization of @c vecmem::details::tuple_get_impl for the 0th element
template <>
struct tuple_get_impl<0> {

    /// Get the first (const) tuple element
    template <typename... Ts>
    VECMEM_HOST_AND_DEVICE static constexpr const auto &get(
        const tuple<Ts...> &t) {
        return t.m_head;
    }
    /// Get the first (non-const) tuple element
    template <typename... Ts>
    VECMEM_HOST_AND_DEVICE static constexpr auto &get(tuple<Ts...> &t) {
        return t.m_head;
    }

};  // struct tuple_get_impl

}  // namespace details

template <std::size_t I, typename... Ts>
VECMEM_HOST_AND_DEVICE inline constexpr const auto &get(
    const tuple<Ts...> &t) noexcept {

    // Make sure that the requested index is valid.
    static_assert(I < sizeof...(Ts),
                  "Attempt to access index greater than tuple size.");

    // Return the correct element using the helper struct.
    return details::tuple_get_impl<I>::get(t);
}

template <std::size_t I, typename... Ts>
VECMEM_HOST_AND_DEVICE inline constexpr auto &get(tuple<Ts...> &t) noexcept {

    // Make sure that the requested index is valid.
    static_assert(I < sizeof...(Ts),
                  "Attempt to access index greater than tuple size.");

    // Return the correct element using the helper struct.
    return details::tuple_get_impl<I>::get(t);
}

template <typename... Ts>
VECMEM_HOST_AND_DEVICE inline constexpr tuple<Ts &...> tie(Ts &... args) {

    return tuple<Ts &...>(args...);
}

template <class... Ts>
VECMEM_HOST_AND_DEVICE inline constexpr tuple<typename std::decay<Ts>::type...>
make_tuple(Ts &&... args) {
    return tuple<typename std::decay<Ts>::type...>{std::forward<Ts>(args)...};
}

}  // namespace vecmem
