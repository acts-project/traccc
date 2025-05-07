/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "kalman_fitting_telescope_test.hpp"
#include "traccc/finding/finding_config.hpp"

namespace traccc {

/// Combinatorial Kalman Finding Test with Sparse tracks
class CkfSparseTrackTelescopeTests : public KalmanFittingTelescopeTests {
    public:
    using ckf_navigator_type =
        detray::navigator<const device_detector_type,
                          traccc::detail::ckf_nav_cache_size>;
};

/// Combinatorial Kalman Finding Test with Identical tracks
class CkfCombinatoricsTelescopeTests : public KalmanFittingTelescopeTests {
    public:
    using ckf_navigator_type =
        detray::navigator<const device_detector_type,
                          traccc::detail::ckf_nav_cache_size>;
};

/// Combinatorial Kalman Finding Test with Identical tracks (CPU)
class CpuCkfCombinatoricsTelescopeTests
    : public CkfCombinatoricsTelescopeTests {};

/// Combinatorial Kalman Finding Test with Identical tracks (CUDA)
class CudaCkfCombinatoricsTelescopeTests
    : public CkfCombinatoricsTelescopeTests {};

}  // namespace traccc
