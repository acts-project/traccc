/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "kalman_fitting_test.hpp"

namespace traccc {

/// Combinatorial Kalman Finding Test with Sparse tracks
class CkfSparseTrackTests : public KalmanFittingTests {};

/// Combinatorial Kalman Finding Test with Dense tracks
class CkfDenseTrackTests : public KalmanFittingTests {};

}  // namespace traccc