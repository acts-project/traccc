/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/utils/type_traits.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <type_traits>

namespace vecmem {

/// Default tuple type
///
/// Serving as the final node in the recursive implementation of this tuple
/// type.
///
template <typename... Ts>
struct tuple {

    // As long as we did everything correctly, this should only get instantiated
    // with an empty parameter list, for the implementation to work correctly.
    static_assert(sizeof...(Ts) == 0,
                  "There's a coding error in vecmem::tuple!");

    /// Default constructor for the default tuple type
    constexpr tuple() = default;

};  // struct tuple

/// Simple tuple implementation for the vecmem EDM classes
///
/// The vecmem EDM classes require something analogous to @c std::tuple,
/// but that type is not officially supported by CUDA in device code. Worse yet,
/// @c std::tuple actively generates invalid code with @c nvcc at the time of
/// writing (up to CUDA 12.3.0).
///
/// This is a very simple implementation for a tuple type, which can do exactly
/// as much as we need from it.
///
/// @tparam T     The first type to be stored in the tuple
/// @tparam ...Ts The rest of the types to be stored in the tuple
///
template <typename T, typename... Ts>
struct tuple<T, Ts...> {

    /// Default constructor
    constexpr tuple() = default;

    /// Copy constructor
    ///
    /// @param parent The parent to copy
    ///
    template <typename U, typename... Us,
              std::enable_if_t<sizeof...(Ts) == sizeof...(Us), bool> = true>
    VECMEM_HOST_AND_DEVICE constexpr tuple(const tuple<U, Us...> &parent)
        : m_head(parent.m_head), m_tail(parent.m_tail) {}

    /// Main constructor, from a list of tuple elements
    ///
    /// @param head The first element to be stored in the tuple
    /// @param tail The rest of the elements to be stored in the tuple
    ///
    template <
        typename U, typename... Us,
        std::enable_if_t<
            vecmem::details::conjunction<
                vecmem::details::negation<std::is_same<tuple<T, Ts...>, U>>,
                std::is_constructible<T, U &&>,
                std::is_constructible<Ts, Us &&>...>::value,
            bool> = true>
    VECMEM_HOST_AND_DEVICE constexpr tuple(U &&head, Us &&... tail)
        : m_head(std::forward<U>(head)), m_tail(std::forward<Us>(tail)...) {}

    /// "Concatenation" constructor
    ///
    /// It is used in the @c vecmem::edm code while constructing some of the
    /// internal tuples of the objects.
    ///
    /// @param head The first element to be stored in the tuple
    /// @param tail The rest of the elements to be stored in the tuple
    ///
    template <typename U, typename... Us,
              std::enable_if_t<vecmem::details::conjunction<
                                   std::is_constructible<T, U &&>,
                                   std::is_constructible<Ts, Us &&>...>::value,
                               bool> = true>
    VECMEM_HOST_AND_DEVICE constexpr tuple(U &&head, tuple<Us...> &&tail)
        : m_head(std::forward<U>(head)), m_tail(std::move(tail)) {}

    /// The first/head element of the tuple
    T m_head;
    /// The rest of the tuple elements
    tuple<Ts...> m_tail;

};  // struct tuple

/// @name Utility functions for @c vecmem::tuple
/// @{

/// Get a constant element out of a tuple
///
/// @tparam I The index of the element to get
/// @tparam ...Ts The types held by the tuple
/// @param t The tuple to get the element from
/// @return The I-th element of the tuple
///
template <std::size_t I, typename... Ts>
VECMEM_HOST_AND_DEVICE inline constexpr const auto &get(
    const tuple<Ts...> &t) noexcept;

/// Get a non-constant element out of a tuple
///
/// @tparam I The index of the element to get
/// @tparam ...Ts The types held by the tuple
/// @param t The tuple to get the element from
/// @return The I-th element of the tuple
///
template <std::size_t I, typename... Ts>
VECMEM_HOST_AND_DEVICE inline constexpr auto &get(tuple<Ts...> &t) noexcept;

/// Tie references to existing objects, into a tuple
///
/// @tparam ...Ts Types to refer to with the resulting tuple
/// @param ...args References to the objects that the tuple should point to
/// @return A tuple of references to some existing objects
///
template <typename... Ts>
VECMEM_HOST_AND_DEVICE inline constexpr tuple<Ts &...> tie(Ts &... args);

/// Make a tuple with automatic type deduction
///
/// @tparam ...Ts Types deduced for the resulting tuple
/// @param args   Values to make a tuple out of
/// @return A tuple constructed from the received values
///
template <class... Ts>
VECMEM_HOST_AND_DEVICE inline constexpr tuple<typename std::decay<Ts>::type...>
make_tuple(Ts &&... args);

/// Default/empty implementation for @c vecmem::tuple_element
///
/// @tparam T Dummy template argument
/// @tparam I Dummy index argument
///
template <std::size_t I, class T>
struct tuple_element;

/// Get the type of the I-th element of a tuple
///
/// @tparam ...Ts The element types in the tuple
/// @tparam I     Index of the element to get the type of
///
template <std::size_t I, typename... Ts>
struct tuple_element<I, tuple<Ts...>> {

    /// Type of the I-th element of the specified tuple
    using type = std::decay_t<decltype(get<I>(std::declval<tuple<Ts...>>()))>;
};

/// Convenience accessor for the I-th element of a tuple
///
/// @tparam T The type of the tuple to investigate
/// @tparam I Index of the element to get the type of
///
template <std::size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;

/// @}

}  // namespace vecmem

// Include the implementation.
#include "vecmem/utils/impl/tuple.ipp"
