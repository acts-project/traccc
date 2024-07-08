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
    typename detector_t::view_type det_data,
    measurement_collection_types::const_view measurements_view,
    bound_track_parameters_collection_types::const_view in_params_view,
    vecmem::data::vector_view<const unsigned int>
        n_measurements_prefix_sum_view,
    vecmem::data::vector_view<const unsigned int> ref_meas_idx_view,
    vecmem::data::vector_view<const candidate_link> prev_links_view,
    vecmem::data::vector_view<const unsigned int> prev_param_to_link_view,
    const unsigned int step, const unsigned int& n_max_candidates,
    bound_track_parameters_collection_types::view out_params_view,
    vecmem::data::vector_view<unsigned int> n_candidates_view,
    vecmem::data::vector_view<candidate_link> links_view,
    unsigned int& n_total_candidates) {

    // Detector
    detector_t det(det_data);

    // Measurement
    measurement_collection_types::const_device measurements(measurements_view);

    // Input parameters
    bound_track_parameters_collection_types::const_device in_params(
        in_params_view);

    // Previous links
    vecmem::device_vector<const candidate_link> prev_links(prev_links_view);

    // Previous param_to_link
    vecmem::device_vector<const unsigned int> prev_param_to_link(
        prev_param_to_link_view);

    // Output parameters
    bound_track_parameters_collection_types::device out_params(out_params_view);

    // Number of candidates per parameter
    vecmem::device_vector<unsigned int> n_candidates(n_candidates_view);

    // Links
    vecmem::device_vector<candidate_link> links(links_view);

    // Prefix sum of the number of measurements per parameter
    vecmem::device_vector<const unsigned int> n_measurements_prefix_sum(
        n_measurements_prefix_sum_view);

    // Reference (first) measurement index per parameter
    vecmem::device_vector<const unsigned int> ref_meas_idx(ref_meas_idx_view);

    // Last step ID
    const int previous_step =
        (step == 0) ? std::numeric_limits<int>::max() : step - 1;

    const unsigned int n_measurements_sum = n_measurements_prefix_sum.back();
    const unsigned int stride = globalIndex * cfg.n_measurements_per_thread;

    vecmem::device_vector<const unsigned int>::iterator lo1;

    for (unsigned int i_meas = 0; i_meas < cfg.n_measurements_per_thread;
         i_meas++) {
        const unsigned int idx = stride + i_meas;

        if (idx >= n_measurements_sum) {
            break;
        }

        if (i_meas == 0 || idx == *lo1) {
            lo1 = thrust::lower_bound(thrust::seq,
                                      n_measurements_prefix_sum.begin(),
                                      n_measurements_prefix_sum.end(), idx + 1);
        }

        const unsigned int in_param_id =
            std::distance(n_measurements_prefix_sum.begin(), lo1);
        const detray::geometry::barcode bcd =
            in_params.at(in_param_id).surface_link();
        const unsigned int offset =
            lo1 == n_measurements_prefix_sum.begin() ? idx : idx - *(lo1 - 1);
        const unsigned int meas_idx = ref_meas_idx.at(in_param_id) + offset;
        bound_track_parameters in_par = in_params.at(in_param_id);

        const auto& meas = measurements.at(meas_idx);
        track_state<typename detector_t::algebra_type> trk_state(meas);
        const detray::tracking_surface sf{det, bcd};

        // Run the Kalman update
        sf.template visit_mask<
            gain_matrix_updater<typename detector_t::algebra_type>>(trk_state,
                                                                    in_par);
        // Get the chi-square
        const auto chi2 = trk_state.filtered_chi2();

        if (chi2 < cfg.chi2_max) {

            // Add measurement candidates to link
            vecmem::device_atomic_ref<unsigned int> num_total_candidates(
                n_total_candidates);

            const unsigned int l_pos = num_total_candidates.fetch_add(1);

            if (l_pos >= n_max_candidates) {
                n_total_candidates = n_max_candidates;
                return;
            }

            // Seed id
            unsigned int orig_param_id =
                (step == 0
                     ? in_param_id
                     : prev_links[prev_param_to_link[in_param_id]].seed_idx);
            // Skip counter
            unsigned int skip_counter =
                (step == 0
                     ? 0
                     : prev_links[prev_param_to_link[in_param_id]].n_skipped);

            links[l_pos] = {{previous_step, in_param_id},
                            meas_idx,
                            orig_param_id,
                            skip_counter};

            // Increase the number of candidates (or branches) per input
            // parameter
            vecmem::device_atomic_ref<unsigned int> num_candidates(
                n_candidates[in_param_id]);
            num_candidates.fetch_add(1);

            out_params[l_pos] = trk_state.filtered();
        }
    }
}

}  // namespace traccc::device
