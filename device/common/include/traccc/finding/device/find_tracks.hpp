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
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_parameters.hpp"

// Thrust include(s)
#include <thrust/binary_search.h>

namespace traccc::device {

/// Function for combinatorial finding.
/// If the chi2 of the measurement < chi2_max, its measurement index and the
/// index of the link from the previous step are added to the link container.
///
/// @param[in] thread_id          A thread identifier object
/// @param[in] barrier            A block-wide barrier
/// @param[in] cfg                Track finding config object
/// @param[in] det_data           Detector view object
/// @param[in] measurements_view  Measurements container view
/// @param[in] in_params_view     Input parameters
/// @param[in] n_in_params        The number of input params
/// @param[in] barcodes_view      View of a measurement -> barcode map
/// @param[in] upper_bounds_view  Upper bounds of measurements unique w.r.t
/// barcode
/// @param[in] prev_links_view    link container from the previous step
/// @param[in] prev_param_to_link_view  param_to_link container from the
/// previous step
/// @param[in] step               Step index
/// @param[in] n_max_candidates   Number of maximum candidates
/// @param[out] out_params_view   Output parameters
/// @param[out] links_view        link container for the current step
/// @param[out] n_total_candidates  The number of total candidates for the
/// current step
/// @param shared_num_candidates  Shared memory scratch space
/// @param shared_candidates      Shared memory scratch space
/// @param shared_candidates_size Shared memory scratch space
///
template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t,
          typename detector_t, typename config_t>
TRACCC_DEVICE inline void find_tracks(
    thread_id_t& thread_id, barrier_t& barrier, const config_t cfg,
    typename detector_t::view_type det_data,
    measurement_collection_types::const_view measurements_view,
    bound_track_parameters_collection_types::const_view in_params_view,
    const unsigned int n_in_params,
    vecmem::data::vector_view<const detray::geometry::barcode> barcodes_view,
    vecmem::data::vector_view<const unsigned int> upper_bounds_view,
    vecmem::data::vector_view<const candidate_link> prev_links_view,
    vecmem::data::vector_view<const unsigned int> prev_param_to_link_view,
    const unsigned int step, const unsigned int& n_max_candidates,
    bound_track_parameters_collection_types::view out_params_view,
    vecmem::data::vector_view<candidate_link> links_view,
    unsigned int& n_total_candidates, unsigned int* shared_num_candidates,
    std::pair<unsigned int, unsigned int>* shared_candidates,
    unsigned int& shared_candidates_size);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/finding/device/impl/find_tracks.ipp"
