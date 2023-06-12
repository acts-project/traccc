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

namespace traccc::device {

/// Function for propagating the kalman-updated tracks to the next surface
///
/// If a track finds a surface that contains measurements, its bound track
/// parameter on the surface will be used for the next step. Otherwise, the link
/// is added into the tip link container so that we can know which links in the
/// link container are the final measurements of full tracks
///
/// @param[in] globalIndex        The index of the current thread
/// @param[in] cfg                Track finding config object
/// @param[in] det_data           Detector view object
/// @param[in] nav_candidates_buffer Navgation buffer
/// @param[in] in_params_view     Input parameters
/// @param[in] links_view         Link container for the current step
/// @param[in] step               Step index
/// @param[in] n_in_params        The number of input parameters
/// @param[out] out_params_view   Output parameters
/// @param[out] param_to_link_view  Container for param index -> link index
/// @param[out] tips_view         Tip link container for the current step
/// @param[out] n_out_params      The number of output parameters
///
template <typename propagator_t, typename config_t>
TRACCC_DEVICE inline void propagate_to_next_surface(
    std::size_t globalIndex, const config_t cfg,
    typename propagator_t::detector_type::detector_view_type det_data,
    vecmem::data::jagged_vector_view<typename propagator_t::intersection_type>
        nav_candidates_buffer,
    bound_track_parameters_collection_types::const_view in_params_view,
    vecmem::data::vector_view<const candidate_link> links_view,
    const unsigned int step, const unsigned int& n_in_params,
    bound_track_parameters_collection_types::view out_params_view,
    vecmem::data::vector_view<unsigned int> param_to_link_view,
    vecmem::data::vector_view<typename candidate_link::link_index_type>
        tips_view,
    unsigned int& n_out_params);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/finding/device/impl/propagate_to_next_surface.ipp"