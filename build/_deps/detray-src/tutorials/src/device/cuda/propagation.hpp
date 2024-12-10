/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/core/detector.hpp"
#include "detray/definitions/units.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/detectors/toy_metadata.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"
#include "detray/tracks/tracks.hpp"
#include "detray/tutorial/types.hpp"

// Covfie include(s)
#include <covfie/cuda/backend/primitive/cuda_device_array.hpp>

namespace detray::tutorial {

namespace bfield::cuda {

// Inhomogeneous field (cuda)
using inhom_bknd_t =
    covfie::backend::affine<covfie::backend::linear<covfie::backend::strided<
        covfie::vector::vector_d<std::size_t, 3>,
        covfie::backend::cuda_device_array<
            covfie::vector::vector_d<detray::scalar, 3>>>>>;

}  // namespace bfield::cuda

// Detector
using detector_host_t = detector<detray::toy_metadata, host_container_types>;
using detector_device_t =
    detector<detray::toy_metadata, device_container_types>;

// Navigator
using navigator_t = navigator<detector_device_t>;
using intersection_t = navigator_t::intersection_type;

// Stepper
using host_field_t = covfie::field<detray::bfield::inhom_bknd_t>;
using device_field_t =
    covfie::field<detray::tutorial::bfield::cuda::inhom_bknd_t>;
using stepper_t =
    rk_stepper<device_field_t::view_t, detray::tutorial::algebra_t>;

// Actors
using actor_chain_t =
    actor_chain<tuple, pathlimit_aborter,
                parameter_transporter<detray::tutorial::algebra_t>,
                pointwise_material_interactor<detray::tutorial::algebra_t>,
                parameter_resetter<detray::tutorial::algebra_t>>;

// Propagator
using propagator_t = propagator<stepper_t, navigator_t, actor_chain_t>;

/// Propagation tutorial function
void propagation(typename detector_host_t::view_type det_data,
                 typename device_field_t::view_t field_data,
                 const vecmem::data::vector_view<
                     free_track_parameters<detray::tutorial::algebra_t>>
                     tracks_data);

}  // namespace detray::tutorial
