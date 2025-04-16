/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/finding/actors/ckf_aborter.hpp"
#include "traccc/finding/actors/interaction_register.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/detector_type_utils.hpp"

// Detray include(s)
#include <detray/propagator/actors.hpp>
#include <detray/propagator/propagator.hpp>
#include <detray/propagator/rk_stepper.hpp>

namespace traccc::cuda::kernels {
using default_finding_algorithm =
    finding_algorithm<stepper_for_t<traccc::default_detector::device>,
                      navigator_for_t<traccc::default_detector::device>>;

}  // namespace traccc::cuda::kernels
