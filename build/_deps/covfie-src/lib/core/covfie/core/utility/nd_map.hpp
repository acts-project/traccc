/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <functional>
#include <utility>

#include <covfie/core/array.hpp>

namespace covfie::utility {
template <typename T, std::size_t N, std::size_t... Ns>
auto tail_impl(
    std::index_sequence<Ns...>, [[maybe_unused]] const array::array<T, N> & t
)
{
    return array::array<T, N - 1>{t.at(Ns + 1u)...};
}

template <typename T, std::size_t N>
auto tail(const array::array<T, N> & t)
{
    return tail_impl(std::make_index_sequence<N - 1u>(), t);
}

template <
    typename T,
    std::size_t N1,
    std::size_t N2,
    std::size_t... Is1,
    std::size_t... Is2>
array::array<T, N1 + N2>
cat_impl(const array::array<T, N1> & a1, const array::array<T, N2> & a2, std::index_sequence<Is1...>, std::index_sequence<Is2...>)
{
    return {a1.at(Is1)..., a2.at(Is2)...};
}

template <typename T, std::size_t N1, std::size_t N2>
array::array<T, N1 + N2>
cat(const array::array<T, N1> & a1, const array::array<T, N2> & a2)
{
    return cat_impl(
        a1, a2, std::make_index_sequence<N1>(), std::make_index_sequence<N2>()
    );
}

template <typename Tuple>
void nd_map(std::function<void(Tuple)> f, Tuple s)
{
    if constexpr (Tuple::dimensions == 0u) {
        f({});
    } else if constexpr (Tuple::dimensions == 1u) {
        for (typename Tuple::value_type i =
                 static_cast<typename Tuple::value_type>(0);
             i < s.at(0);
             ++i)
        {
            f(array::array<typename Tuple::value_type, 1>{i});
        }
    } else {
        using tail_t = decltype(tail(std::declval<Tuple>()));

        for (typename Tuple::value_type i =
                 static_cast<typename Tuple::value_type>(0);
             i < s.at(0);
             ++i)
        {
            nd_map<tail_t>(
                [f, i](tail_t r) {
                    f(cat(array::array<typename Tuple::value_type, 1>{i}, r));
                },
                tail(s)
            );
        }
    }
}
}
