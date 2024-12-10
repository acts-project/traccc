/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/utils/tuple.hpp"
#include "detray/utils/type_traits.hpp"

// System include(s)
#include <string>
#include <string_view>
#include <type_traits>

namespace detray::types {

/// @brief type list implementation
/// @see https://www.codingwiththomas.com/blog/getting-started-with-typelists
template <typename... Ts>
struct list {};

/// Number of types in the list
/// @{
template <typename = void>
struct get_size {};

template <typename... Ts>
struct get_size<list<Ts...>>
    : std::integral_constant<std::size_t, sizeof...(Ts)> {};

template <typename L>
constexpr inline std::size_t size{get_size<L>()};
/// @}

/// Access the first type
/// @{
template <typename = void>
struct get_front {};

template <typename T, typename... Ts>
struct get_front<list<T, Ts...>> {
    using type = T;
};
template <typename L>
using front = typename get_front<L>::type;
/// @}

/// Access the last type
/// @{
template <typename = void>
struct get_back {};

template <typename T, typename... Ts>
struct get_back<list<T, Ts...>> {
    using type = std::conditional_t<sizeof...(Ts) == 0, T,
                                    typename get_back<list<Ts...>>::type>;
};
// Base case
template <>
struct get_back<list<>> {
    using type = void;
};
template <typename L>
using back = typename get_back<L>::type;
/// @}

/// Access the N-th type
/// @{
template <std::size_t N, typename = void>
struct get_at {};

template <std::size_t N, typename T, typename... Ts>
struct get_at<N, list<T, Ts...>> {
    using type = std::remove_cvref_t<decltype(
        detray::get<N>(detray::tuple<T, Ts...>{}))>;
};
template <typename L, std::size_t N>
using at = typename get_at<N, L>::type;
/// @}

/// Append a type
/// @{
template <typename N, typename = void>
struct do_push_back {};

template <typename N, typename... Ts>
struct do_push_back<N, list<Ts...>> {
    using type = list<Ts..., N>;
};
template <typename L, typename N>
using push_back = typename do_push_back<N, L>::type;
/// @}

/// Prepend a type
/// @{
template <typename N, typename = void>
struct do_push_front {};

template <typename N, typename... Ts>
struct do_push_front<N, list<Ts...>> {
    using type = list<N, Ts...>;
};
template <typename L, typename N>
using push_front = typename do_push_front<N, L>::type;
/// @}

/// Print the type list
/// @{

/// @see
/// https://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname
template <typename T>
std::string demangle_type_name() {
#if defined(__clang__)
    constexpr std::string_view prefix{"[T = "};
    constexpr std::string_view suffix{"]"};
    constexpr std::string_view function{__PRETTY_FUNCTION__};
#elif defined(__GNUC__)
    constexpr std::string_view prefix{"with T = "};
    constexpr std::string_view suffix{"; "};
    constexpr std::string_view function{__PRETTY_FUNCTION__};
#elif defined(_MSC_VER)
    constexpr std::string_view prefix{"get_type_name<"};
    constexpr std::string_view suffix{">(void)"};
    constexpr std::string_view function{__FUNCSIG__};
#else
    std::string type_name{typeid(T).name()};
    type_name += " ";
    std::string_view function{type_name};
    constexpr std::string_view prefix{""};
    constexpr std::string_view suffix{" "};
#endif

    const std::size_t start{function.find(prefix) + prefix.size()};
    const std::size_t end{function.find(suffix)};
    const std::size_t size{end - start};

    return std::string{function.substr(start, size)};
}

template <typename = void>
struct print {};

template <typename... Ts>
struct print<list<Ts...>> {

    template <typename P = void, typename... Ps>
    void print_typeid() {

        std::printf("%s", demangle_type_name<P>().c_str());

        // Keep unrolling the pack
        if constexpr (sizeof...(Ps) > 0) {
            std::printf(", ");
            return print_typeid<Ps...>();
        }
    }

    print() {
        std::printf("type_list<");
        print_typeid<Ts...>();
        std::printf(">\n");
    }
};
/// @}

}  // namespace detray::types
