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

// Detray include(s)
#include <detray/detectors/bfield.hpp>
#include <detray/propagator/actors.hpp>
#include <detray/propagator/propagator.hpp>
#include <detray/propagator/rk_stepper.hpp>

namespace traccc::cuda::kernels {

using default_detector_type = traccc::default_detector::device;
using default_stepper_type = detray::rk_stepper<
    covfie::field<detray::bfield::const_bknd_t<
        default_detector_type::scalar_type>>::view_t,
    default_detector_type::algebra_type,
    detray::constrained_step<default_detector_type::scalar_type>>;
using default_navigator_type = detray::navigator<const default_detector_type>;

using default_finding_algorithm =
    finding_algorithm<default_stepper_type, default_navigator_type>;

}  // namespace traccc::cuda::kernels
