/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/utils/tuple.hpp"
#include "detray/utils/type_traits.hpp"

// System include(s)
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace detray::detail {

/// get function accessor for std::tuple
///
/// usage example:
/// detail::get<0>(tuple)
/// @{
using std::get;

template <std::size_t I, typename... value_types>
DETRAY_HOST_DEVICE constexpr decltype(auto) get(
    const ::detray::tuple<value_types...>& tuple) noexcept {
    return ::detray::get<I>(tuple);
}

template <std::size_t I, typename... value_types>
DETRAY_HOST_DEVICE constexpr decltype(auto) get(
    ::detray::tuple<value_types...>& tuple) noexcept {
    return ::detray::get<I>(tuple);
}

template <typename query_t, typename... value_types>
DETRAY_HOST_DEVICE constexpr decltype(auto) get(
    const ::detray::tuple<value_types...>& tuple) noexcept {
    return ::detray::get<get_type_pos_v<query_t, value_types...>>(tuple);
}

template <typename query_t, typename... value_types>
DETRAY_HOST_DEVICE constexpr decltype(auto) get(
    ::detray::tuple<value_types...>& tuple) noexcept {
    return ::detray::get<get_type_pos_v<query_t, value_types...>>(tuple);
}
/// @}

/// tuple_element for std::tuple
///
/// usage example:
/// detail::tuple_element< int, tuple_t >::type
/// @{
template <std::size_t N, class T>
struct tuple_element;

// std::tuple
template <std::size_t N, typename... value_types>
struct tuple_element<N, std::tuple<value_types...>>
    : std::tuple_element<N, std::tuple<value_types...>> {};

// detray::tuple
template <std::size_t N, typename... value_types>
struct tuple_element<N, detray::tuple<value_types...>> {
    using type = std::decay_t<decltype(
        ::detray::get<N>(std::declval<detray::tuple<value_types...>>()))>;
};

template <std::size_t N, class T>
using tuple_element_t = typename tuple_element<N, T>::type;
/// @}

/// tuple_size for std::tuple
///
/// usage example:
/// detail::tuple_size< tuple_t >::value
/// @{
template <class T>
struct tuple_size;

// std::tuple
template <typename... value_types>
struct tuple_size<std::tuple<value_types...>>
    : std::tuple_size<std::tuple<value_types...>> {};

// detray::tuple
template <typename... value_types>
struct tuple_size<::detray::tuple<value_types...>> {
    static constexpr std::size_t value = sizeof...(value_types);
};

template <class T>
constexpr std::size_t tuple_size_v{tuple_size<T>::value};
/// @}

/// make_tuple for std::tuple
/// users have to specifiy tuple_t for detail::make_tuple
///
/// usage example
/// detail::make_tuple<tuple_t>(args...)
/// @{
template <class T>
struct unwrap_refwrapper {
    using type = T;
};

template <class T>
struct unwrap_refwrapper<std::reference_wrapper<T>> {
    using type = T&;
};

template <class T>
using unwrap_decay_t = typename unwrap_refwrapper<std::decay_t<T>>::type;

// make_tuple for std::tuple
template <template <typename...> class tuple_t, class... value_types>
requires std::is_same_v<tuple_t<value_types...>, std::tuple<value_types...>>
    DETRAY_HOST constexpr std::tuple<unwrap_decay_t<value_types>...> make_tuple(
        value_types&&... args) {
    return std::make_tuple(std::forward<value_types>(args)...);
}

// make_tuple for detray::tuple
template <template <typename...> class tuple_t, class... value_types>
requires std::is_same_v<tuple_t<value_types...>, detray::tuple<value_types...>>
    DETRAY_HOST_DEVICE constexpr detray::tuple<unwrap_decay_t<value_types>...>
    make_tuple(value_types&&... args) {
    return detray::tuple<unwrap_decay_t<value_types>...>{
        std::forward<value_types>(args)...};
}
/// @}

/// Check if the tuple contains a type
/// @see
/// https://stackoverflow.com/questions/25958259/how-do-i-find-out-if-a-tuple-contains-a-type
/// @{
template <typename T, typename tuple_t>
struct has_type;

// std::tuple
template <typename T>
struct has_type<T, std::tuple<>> : std::false_type {};

template <typename T, typename U, typename... Ts>
struct has_type<T, std::tuple<U, Ts...>> : has_type<T, std::tuple<Ts...>> {};

template <typename T, typename... Ts>
struct has_type<T, std::tuple<T, Ts...>> : std::true_type {};

// detray::tuple
template <typename T>
struct has_type<T, detray::tuple<>> : std::false_type {};

template <typename T, typename U, typename... Ts>
struct has_type<T, detray::tuple<U, Ts...>>
    : has_type<T, detray::tuple<Ts...>> {};

template <typename T, typename... Ts>
struct has_type<T, detray::tuple<T, Ts...>> : std::true_type {};

template <typename T, class tuple_t>
constexpr bool has_type_v = has_type<T, tuple_t>::value;
///@}

}  // namespace detray::detail
