/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "detray/detectors/bfield.hpp"
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/finding/actors/ckf_aborter.hpp"
#include "traccc/finding/actors/interaction_register.hpp"
#include "traccc/geometry/detector.hpp"

namespace traccc::cuda::kernels {

using default_detector_type =
    detray::detector<detray::default_metadata, detray::device_container_types>;
using default_stepper_type =
    detray::rk_stepper<covfie::field<detray::bfield::const_bknd_t>::view_t,
                       traccc::default_algebra, detray::constrained_step<>>;
using default_navigator_type = detray::navigator<const default_detector_type>;

using default_finding_algorithm =
    finding_algorithm<default_stepper_type, default_navigator_type>;

}  // namespace traccc::cuda::kernels
