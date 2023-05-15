/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

namespace traccc {

TRACCC_HOST_DEVICE inline vector3 make_direction_unit_from_phi_theta(
    scalar phi, scalar theta) {
    const auto cosTheta = std::cos(theta);
    const auto sinTheta = std::sin(theta);
    return {
        std::cos(phi) * sinTheta,
        std::sin(phi) * sinTheta,
        cosTheta,
    };
}

TRACCC_HOST_DEVICE inline scalar phi(const vector3& v) {
    return std::atan2(v[1], v[0]);
}

TRACCC_HOST_DEVICE inline scalar eta(const vector3& v) {
    return std::atanh(v[2] / getter::norm(v));
}

TRACCC_HOST_DEVICE inline scalar theta(const vector3& v) {
    return std::atan2(std::sqrt(v[0] * v[0] + v[1] * v[1]), v[2]);
}

template <typename range_t>
TRACCC_HOST_DEVICE inline range_t eta_to_theta_range(const range_t& eta_range) {
    // @NOTE: eta_range[0] is converted to theta_range[1] and eta_range[1]
    // to theta_range[0] because theta(minEta) > theta(maxEta)
    return {2 * std::atan(std::exp(-eta_range[1])),
            2 * std::atan(std::exp(-eta_range[0]))};
}

}  // namespace traccc
