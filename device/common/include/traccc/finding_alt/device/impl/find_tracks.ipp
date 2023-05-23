/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

template <typename propagator_t, typename config_t>
TRACCC_DEVICE inline void find_tracks(
    std::size_t globalIndex, const config_t cfg,
    typename propagator_t::detector_type::detector_view_type det_data,
    vecmem::data::jagged_vector_view<typename propagator_t::intersection_type>
        nav_candidates_buffer,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const thrust::pair<geometry_id, unsigned int>>
        module_map_view,
    bound_track_parameters_collection_types::const_view seeds_view,
    vecmem::data::vector_view<candidate_link_alt> links_view,
    vecmem::data::vector_view<typename candidate_link_alt::link_index_type>
        tips_view) {

    /*
    if (globalIndex >= n_in_params) {
        return;
    }
    */

    // Detector
    typename propagator_t::detector_type det(det_data);

    // Navigation candidate buffer
    vecmem::jagged_device_vector<typename propagator_t::intersection_type>
        nav_candidates(nav_candidates_buffer);

    // Measurement
    measurement_container_types::const_device measurements(measurements_view);

    // Module map
    vecmem::device_vector<const thrust::pair<geometry_id, unsigned int>>
        module_map(module_map_view);

    // Seeds
    bound_track_parameters_collection_types::const_device seeds(seeds_view);

    // Links
    vecmem::device_vector<candidate_link_alt> links(links_view);

    // Tips
    vecmem::device_vector<typename candidate_link_alt::link_index_type> tips(
        tips_view);

    for (unsigned int step = 0; step < cfg.max_track_candidates_per_track;
         step++) {
    }
}

}  // namespace traccc::device