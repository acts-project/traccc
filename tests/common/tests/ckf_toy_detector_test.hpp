/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "kalman_fitting_toy_detector_test.hpp"
#include "traccc/finding/finding_config.hpp"

namespace traccc {

/// Combinatorial Kalman Finding Test to Comapre CPU results
class CkfToyDetectorTests : public KalmanFittingToyDetectorTests {
    public:
    using ckf_navigator_type =
        detray::navigator<const device_detector_type,
                          traccc::detail::ckf_nav_cache_size>;
};

}  // namespace traccc
