/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/utils/particle.hpp"

namespace traccc::device {
template <typename propagator_t, typename bfield_t>
struct propagate_to_next_surface_payload {
    typename propagator_t::detector_type::view_type det_data;
    bfield_t field_data;
    bound_track_parameters_collection_types::view params_view;
    vecmem::data::vector_view<unsigned int> params_liveness_view;
    const vecmem::data::vector_view<const unsigned int>& param_ids_view;
    vecmem::data::vector_view<const candidate_link> links_view;
    const unsigned int step;
    const unsigned int n_in_params;
    vecmem::data::vector_view<typename candidate_link::link_index_type>
        tips_view;
    vecmem::data::vector_view<unsigned int> n_tracks_per_seed_view;
};

/// Function for propagating the kalman-updated tracks to the next surface
///
/// If a track finds a surface that contains measurements, its bound track
/// parameter on the surface will be used for the next step. Otherwise, the link
/// is added into the tip link container so that we can know which links in the
/// link container are the final measurements of full tracks
///
/// @param[in] globalIndex        The index of the current thread
/// @param[in] cfg                Track finding config object
/// @param[inout] payload      The function call payload
template <typename propagator_t, typename bfield_t, typename config_t>
TRACCC_DEVICE inline void propagate_to_next_surface(
    std::size_t globalIndex, const config_t cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload);
}  // namespace traccc::device

#include "./impl/propagate_to_next_surface.ipp"
