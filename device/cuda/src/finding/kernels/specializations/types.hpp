/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/utils/bfield.hpp"
#include "traccc/finding/actors/ckf_aborter.hpp"
#include "traccc/finding/actors/interaction_register.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/detector_type_utils.hpp"
#include "traccc/utils/propagation.hpp"

namespace traccc::cuda {
using default_finding_algorithm =
    finding_algorithm<stepper_for_t<traccc::default_detector::device>,
                      navigator_for_t<traccc::default_detector::device>>;

template <typename detector_t>
using inhom_stepper_for_t = ::detray::rk_stepper<
    typename ::covfie::field<
        cuda::inhom_bfield_backend_t<typename detector_t::scalar_type>>::view_t,
    typename detector_t::algebra_type,
    ::detray::constrained_step<typename detector_t::scalar_type>>;

using inhomogeneous_field_default_finding_algorithm =
    finding_algorithm<inhom_stepper_for_t<traccc::default_detector::device>,
                      navigator_for_t<traccc::default_detector::device>>;

}  // namespace traccc::cuda
