/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"

// System include(s).
#include <limits>

namespace traccc::device {

template <typename detector_t, typename config_t>
TRACCC_DEVICE inline void find_tracks(
    std::size_t globalIndex, const config_t cfg,
    typename detector_t::detector_view_type det_data,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const thrust::pair<geometry_id, unsigned int>>
        module_map_view,
    bound_track_parameters_collection_types::const_view in_params_view,
    vecmem::data::vector_view<const unsigned int> n_threads_view,
    const unsigned int step, const unsigned int& n_measurements_per_thread,
    const unsigned int& n_total_threads,
    bound_track_parameters_collection_types::view out_params_view,
    vecmem::data::vector_view<candidate_link> links_view,
    unsigned int& n_candidates) {

    if (globalIndex >= n_total_threads) {
        return;
    }

    // Detector
    detector_t det(det_data);

    // Measurement
    measurement_container_types::const_device measurements(measurements_view);

    // module map
    vecmem::device_vector<const thrust::pair<geometry_id, unsigned int>>
        module_map(module_map_view);

    // Input parameters
    bound_track_parameters_collection_types::const_device in_params(
        in_params_view);

    // Output parameters
    bound_track_parameters_collection_types::device out_params(out_params_view);

    // Links
    vecmem::device_vector<candidate_link> links(links_view);

    // n threads
    vecmem::device_vector<const unsigned int> n_threads(n_threads_view);

    // Search for out_param index
    const auto lo1 = thrust::lower_bound(thrust::seq, n_threads.begin(),
                                         n_threads.end(), globalIndex + 1);
    const auto in_param_id = std::distance(n_threads.begin(), lo1);

    // Get module id
    const auto module_id = in_params.at(in_param_id).surface_link();

    // Search for measuremetns header ID
    const auto lo2 = thrust::lower_bound(
        thrust::seq, module_map.begin(), module_map.end(), module_id.value(),
        compare_pair_int<thrust::pair, unsigned int>());
    const auto header_id = (*lo2).second;

    // Get measurements on surface
    const auto measurements_on_surface = measurements.at(header_id).items;

    unsigned int ref;
    if (lo1 == n_threads.begin()) {
        ref = 0;
    } else {
        ref = *(lo1 - 1);
    }

    const unsigned int offset = globalIndex - ref;
    const unsigned int stride = offset * n_measurements_per_thread;
    const unsigned int n_meas_on_surface = measurements_on_surface.size();

    // Iterate over the measurements
    const auto& mask_store = det.mask_store();
    const auto& surface = det.surfaces(module_id);

    // Last step ID
    const unsigned int previous_step =
        (step == 0) ? std::numeric_limits<unsigned int>::max() : step - 1;

    for (unsigned int i = 0; i < n_measurements_per_thread; i++) {
        if (i + stride >= n_meas_on_surface) {
            break;
        }

        bound_track_parameters in_par = in_params.at(in_param_id);
        const auto meas = measurements_on_surface.at(i + stride);
        track_state<typename detector_t::transform3> trk_state(
            {module_id, meas});

        // Run the Kalman update
        mask_store.template visit<
            gain_matrix_updater<typename detector_t::transform3>>(
            surface.mask(), trk_state, in_par);

        // Get the chi-square
        const auto chi2 = trk_state.filtered_chi2();

        if (chi2 < cfg.chi2_max) {

            // Add measurement candidates to link
            vecmem::device_atomic_ref<unsigned int> num_candidates(
                n_candidates);
            const unsigned int l_pos = num_candidates.fetch_add(1);

            // @TODO; Consider max_num_branches_per_surface
            links[l_pos] = {{previous_step, in_param_id},
                            {header_id, i + stride},
                            module_id};

            out_params[l_pos] = trk_state.filtered();
        }
    }
}

}  // namespace traccc::device