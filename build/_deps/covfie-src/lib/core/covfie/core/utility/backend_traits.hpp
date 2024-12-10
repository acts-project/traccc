/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstddef>
#include <variant>

namespace covfie::utility {
template <typename B, std::size_t N, bool = B::is_initial>
struct nth_backend {
    using type = typename nth_backend<typename B::backend_t, N - 1>::type;
};

template <typename B>
struct nth_backend<B, 0, false> {
    using type = B;
};

template <typename B>
struct nth_backend<B, 0, true> {
    using type = B;
};

template <typename B, std::size_t N>
struct nth_backend<B, N, true> {
    using type = std::monostate;
};

template <typename B, bool = B::is_initial>
struct backend_depth {
};

template <typename B>
struct backend_depth<B, true> {
    static constexpr std::size_t value = 1;
};

template <typename B>
struct backend_depth<B, false> {
    static constexpr std::size_t value =
        backend_depth<typename B::backend_t>::value + 1;
};
}
