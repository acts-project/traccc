/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/utils/find_bound.hpp"
#include "detray/utils/ranges/detail/iterator_functions.hpp"
#include "detray/utils/sort.hpp"

// System include(s)
#include <algorithm>

namespace detray::detail {

/// @brief sequential (single thread) sort function
template <std::random_access_iterator rand_iter_t,
          std::sentinel_for<rand_iter_t> sentinel_t>
DETRAY_HOST_DEVICE inline void sequential_sort(rand_iter_t first,
                                               sentinel_t last) {
#if defined(__CUDACC__) || defined(CL_SYCL_LANGUAGE_VERSION) || \
    defined(SYCL_LANGUAGE_VERSION)
    detray::selection_sort(first, last);
#else
    std::ranges::sort(first, last);
#endif
}

/// @brief find_if implementation for host/devcie (single thread)
template <std::random_access_iterator rand_iter_t,
          std::sentinel_for<rand_iter_t> sentinel_t, class Predicate>
DETRAY_HOST_DEVICE inline auto find_if(rand_iter_t first, sentinel_t last,
                                       Predicate&& comp) {
    for (rand_iter_t i = first; i != last; ++i) {
        if (std::forward<Predicate>(comp)(*i)) {
            return i;
        }
    }

    return last;
}

/// @brief lower_bound implementation for host/device
template <std::forward_iterator forw_iter_t,
          std::sentinel_for<forw_iter_t> sentinel_t, typename Value>
DETRAY_HOST_DEVICE inline auto lower_bound(forw_iter_t first, sentinel_t last,
                                           const Value& value) {
#if defined(__CUDACC__) || defined(CL_SYCL_LANGUAGE_VERSION) || \
    defined(SYCL_LANGUAGE_VERSION)
    return detray::lower_bound(first, last, value);
#else
    return std::ranges::lower_bound(first, last, value);
#endif
}

/// @brief upper_bound implementation for host/device
template <std::forward_iterator forw_iter_t,
          std::sentinel_for<forw_iter_t> sentinel_t, typename Value>
DETRAY_HOST_DEVICE inline auto upper_bound(forw_iter_t first, sentinel_t last,
                                           const Value& value) {
#if defined(__CUDACC__) || defined(CL_SYCL_LANGUAGE_VERSION) || \
    defined(SYCL_LANGUAGE_VERSION)
    return detray::upper_bound(first, last, value);
#else
    return std::ranges::upper_bound(first, last, value);
#endif
}

}  // namespace detray::detail
