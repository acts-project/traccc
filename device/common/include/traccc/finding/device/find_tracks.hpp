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
#include "traccc/edm/track_state.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"

// Thrust include(s)
#include <thrust/binary_search.h>

namespace traccc::device {
template <typename detector_t>
struct find_tracks_payload {
    typename detector_t::view_type det_data;
    measurement_collection_types::const_view measurements_view;
    bound_track_parameters_collection_types::const_view in_params_view;
    vecmem::data::vector_view<const unsigned int> in_params_liveness_view;
    const unsigned int n_in_params;
    vecmem::data::vector_view<const detray::geometry::barcode> barcodes_view;
    vecmem::data::vector_view<const unsigned int> upper_bounds_view;
    vecmem::data::vector_view<const candidate_link> prev_links_view;
    const unsigned int step;
    const unsigned int n_max_candidates;
    bound_track_parameters_collection_types::view out_params_view;
    vecmem::data::vector_view<unsigned int> out_params_liveness_view;
    vecmem::data::vector_view<candidate_link> links_view;
    unsigned int* n_total_candidates;
};

struct find_tracks_shared_payload {
    unsigned int* shared_num_candidates;
    std::pair<unsigned int, unsigned int>* shared_candidates;
    unsigned int& shared_candidates_size;
};

/// Function for combinatorial finding.
/// If the chi2 of the measurement < chi2_max, its measurement index and the
/// index of the link from the previous step are added to the link container.
///
/// @param[in] thread_id          A thread identifier object
/// @param[in] barrier            A block-wide barrier
/// @param[in] cfg                Track finding config object
/// @param[inout] payload         The global memory payload
/// @param[inout] shared_payload  The shared memory payload
template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t,
          typename detector_t, typename config_t>
TRACCC_DEVICE inline void find_tracks(
    thread_id_t& thread_id, barrier_t& barrier, const config_t cfg,
    const find_tracks_payload<detector_t>& payload,
    const find_tracks_shared_payload& shared_payload);
}  // namespace traccc::device

#include "./impl/find_tracks.ipp"
