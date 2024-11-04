/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/math.hpp"
#include "traccc/definitions/qualifiers.hpp"

// System include(s).
#include <cmath>

namespace traccc {

template <typename range_t>
TRACCC_HOST_DEVICE inline range_t eta_to_theta_range(const range_t& eta_range) {
    // @NOTE: eta_range[0] is converted to theta_range[1] and eta_range[1]
    // to theta_range[0] because theta(minEta) > theta(maxEta)
    return {2.f * math::atan(std::exp(-eta_range[1])),
            2.f * math::atan(std::exp(-eta_range[0]))};
}

template <typename range_t>
TRACCC_HOST_DEVICE inline range_t theta_to_eta_range(
    const range_t& theta_range) {
    // @NOTE: theta_range[0] is converted to eta_range[1] and theta_range[1]
    // to eta_range[0]
    return {-math::log(math::tan(theta_range[1] * 0.5f)),
            -math::log(math::tan(theta_range[0] * 0.5f))};
}

}  // namespace traccc
