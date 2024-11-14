/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/utils/memory_resource.hpp"

// Detray include(s).
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/propagator.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// SYCL include(s).
#include <sycl/sycl.hpp>

namespace traccc::sycl::details {

/// Templated implementation of the track finding algorithm.
///
/// Concrete track finding algorithms can use this function with the appropriate
/// specializations, to find tracks on top of a specific detector type, magnetic
/// field type, and track finding configuration.
///
/// @tparam stepper_t The stepper type used for the track propagation
/// @tparam navigator_t The navigator type used for the track navigation
///
/// @param det               A view of the detector object
/// @param field             The magnetic field object
/// @param measurements_view All measurements in an event
/// @param seeds_view        All seeds in an event to start the track finding
///                          with
/// @param config            The track finding configuration
/// @param mr                The memory resource(s) to use
/// @param copy              The copy object to use
/// @param queue             The SYCL queue to use
///
/// @return A buffer of the found track candidates
///
template <typename stepper_t, typename navigator_t>
track_candidate_container_types::buffer find_tracks(
    const typename navigator_t::detector_type::view_type& /*det*/,
    const typename stepper_t::magnetic_field_type& /*field*/,
    const measurement_collection_types::const_view& /*measurements_view*/,
    const bound_track_parameters_collection_types::const_view& /*seeds_view*/,
    const finding_config& /*config*/, const memory_resource& /*mr*/,
    vecmem::copy& /*copy*/, ::sycl::queue& /*queue*/) {

    return {};
}

}  // namespace traccc::sycl::details
