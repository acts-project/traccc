/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <traccc/definitions/qualifiers.hpp>
#include <type_traits>

namespace traccc {
/**
 * @brief A simple pair type.
 */
template <typename T1, typename T2>
struct pair {
    public:
    using first_type = T1;
    using second_type = T2;

    TRACCC_HOST_DEVICE constexpr pair() {}

    TRACCC_HOST_DEVICE constexpr pair(const T1& _v1, const T2& _v2)
        : first(_v1), second(_v2) {}

    template <typename S1, typename S2,
              std::enable_if_t<std::is_constructible_v<T1, const S1&> &&
                                   std::is_constructible_v<T2, const S2&>,
                               bool> = true>
    TRACCC_HOST_DEVICE constexpr pair(const pair<S1, S2>& p)
        : first(p.first), second(p.second) {}

    T1 first;
    T2 second;
};
}  // namespace traccc
