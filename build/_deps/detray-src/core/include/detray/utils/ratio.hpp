/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s)
#include <ratio>

namespace detray {

/// Helper struct to sum over variadic std::ratio
/// @{
template <typename... ratios>
struct ratio_sum_helper;

template <typename R>
struct ratio_sum_helper<R> {
    using ratio = R;
};

template <typename ratio1, typename ratio2, typename... ratios>
struct ratio_sum_helper<ratio1, ratio2, ratios...> {

    using first_two_sum = std::ratio_add<ratio1, ratio2>;

    using next_helper = ratio_sum_helper<first_two_sum, ratios...>;

    // recursive summation of first two std::ratio
    using ratio =
        typename std::conditional_t<sizeof...(ratios) == 0, first_two_sum,
                                    typename next_helper::ratio>;
};
/// @}

/// Struct to sum over variadic std::ratio
template <typename... ratios>
struct ratio_sum {
    using helper = ratio_sum_helper<ratios...>;
    using ratio = typename helper::ratio;
};

/// Helper trait to check if the ratio is one
template <typename R>
struct is_ratio_one {
    static constexpr bool value = (R::num == R::den);
};

template <class R>
inline constexpr bool is_ratio_one_v = is_ratio_one<R>::value;

}  // namespace detray
