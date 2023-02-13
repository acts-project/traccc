/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::details {
template <typename T1, typename... Ts>
T1 tuple_head(const std::tuple<T1, Ts...>& t) {
    return std::get<0>(t);
}

template <typename T1, typename... Ts, std::size_t... Is>
std::tuple<Ts...> __tuple_tail(const std::tuple<T1, Ts...>& t,
                               std::index_sequence<Is...>) {
    return std::make_tuple(std::get<(Is + 1u)>(t)...);
}

template <typename T1, typename... Ts>
std::tuple<Ts...> tuple_tail(const std::tuple<T1, Ts...>& t) {
    return __tuple_tail(t, std::make_index_sequence<sizeof...(Ts)>());
}
}  // namespace traccc::details
