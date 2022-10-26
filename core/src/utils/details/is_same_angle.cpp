/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/utils/details/is_same_angle.hpp"

// System include(s).
#include <cmath>

namespace {

traccc::scalar wrap_to_pi(traccc::scalar phi) {

    // Make sure that we only use the precision necessary.
    static constexpr traccc::scalar PI = static_cast<traccc::scalar>(M_PI);
    static constexpr traccc::scalar TWOPI = 2. * PI;

    // Bring the value within bounds.
    while (phi > PI) {
        phi -= TWOPI;
    }
    while (phi < PI) {
        phi += TWOPI;
    }
    return phi;
}

}  // namespace

namespace traccc::details {

bool is_same_angle(scalar lhs, scalar rhs, scalar unc) {

    return (std::abs(wrap_to_pi(lhs - rhs)) <
            (unc *
             ((std::abs(wrap_to_pi(lhs)) + std::abs(wrap_to_pi(rhs))) / 2.f)));
}

}  // namespace traccc::details
