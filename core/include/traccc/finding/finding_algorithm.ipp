/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/finding/candidate_link.hpp"

// detray include(s).
#include "detray/geometry/barcode.hpp"
#include "detray/geometry/surface.hpp"
#include "detray/navigation/detail/ray.hpp"
#include "detray/navigation/intersection/ray_intersector.hpp"
#include "detray/navigation/intersection_kernel.hpp"

// System include
#include <algorithm>
#include <limits>

namespace traccc {

template <typename stepper_t, typename navigator_t>
track_candidate_container_types::host
finding_algorithm<stepper_t, navigator_t>::operator()(
    const detector_type& det, const bfield_type& field,
    const measurement_collection_types::host& measurements,
    const bound_track_parameters_collection_types::host& seeds) const {

    track_candidate_container_types::host output_candidates;

    /*****************************************************************
     * Measurement Operations
     *****************************************************************/

    // Get copy of barcode uniques
    std::vector<measurement> uniques;
    uniques.resize(measurements.size());

    auto end = std::unique_copy(measurements.begin(), measurements.end(),
                                uniques.begin(), measurement_equal_comp());
    const unsigned int n_modules = end - uniques.begin();

    // Get upper bounds of unique elements
    std::vector<unsigned int> upper_bounds;
    upper_bounds.reserve(n_modules);
    for (unsigned int i = 0; i < n_modules; i++) {
        auto up = std::upper_bound(measurements.begin(), measurements.end(),
                                   uniques[i], measurement_sort_comp());
        upper_bounds.push_back(std::distance(measurements.begin(), up));
    }

    // Get the number of measurements of each module
    std::vector<unsigned int> sizes(n_modules);
    std::adjacent_difference(upper_bounds.begin(), upper_bounds.end(),
                             sizes.begin());

    // Create barcode sequence
    std::vector<detray::geometry::barcode> barcodes;
    barcodes.reserve(n_modules);
    for (unsigned int i = 0; i < n_modules; i++) {
        barcodes.push_back(uniques[i].surface_link);
    }

    /**********************
     * Find tracks
     **********************/

    std::vector<std::vector<candidate_link>> links;
    links.resize(m_cfg.max_track_candidates_per_track);

    std::vector<std::vector<std::size_t>> param_to_link;
    param_to_link.resize(m_cfg.max_track_candidates_per_track);

    std::vector<typename candidate_link::link_index_type> tips;

    // Create propagator
    propagator_type propagator(m_cfg.propagation);

    // Copy seed to input parameters
    std::vector<bound_track_parameters> in_params;
    std::vector<unsigned int> n_trks_per_seed(seeds.size(), 0);

    in_params.reserve(seeds.size());
    for (const auto& seed : seeds) {
        in_params.push_back(seed);
    }

    std::vector<bound_track_parameters> out_params;

    for (unsigned int step = 0; step < m_cfg.max_track_candidates_per_track;
         step++) {

        // Iterate over input parameters
        const std::size_t n_in_params = in_params.size();

        // Terminate if there is no parameter to proceed
        if (n_in_params == 0) {
            break;
        }

        // Rough estimation on out parameters size
        out_params.reserve(n_in_params);

        // Previous step ID
        const unsigned int previous_step =
            (step == 0) ? std::numeric_limits<unsigned int>::max() : step - 1;

        std::fill(n_trks_per_seed.begin(), n_trks_per_seed.end(), 0);

        for (unsigned int in_param_id = 0; in_param_id < n_in_params;
             in_param_id++) {

            bound_track_parameters& in_param = in_params[in_param_id];
            unsigned int orig_param_id =
                (step == 0
                     ? in_param_id
                     : links[step - 1][param_to_link[step - 1][in_param_id]]
                           .seed_idx);
            unsigned int skip_counter =
                (step == 0
                     ? 0
                     : links[step - 1][param_to_link[step - 1][in_param_id]]
                           .n_skipped);

            /*************************
             * Material interaction
             *************************/

            // Get intersection at surface
            const detray::surface<detector_type> sf{det,
                                                    in_param.surface_link()};

            const cxt_t ctx{};
            const auto free_vec =
                sf.bound_to_free_vector(ctx, in_param.vector());
            intersection_type sfi;

            const auto sf_desc = det.surface(in_param.surface_link());
            sfi.sf_desc = sf_desc;
            sf.template visit_mask<
                detray::intersection_update<detray::ray_intersector>>(
                detray::detail::ray<transform3_type>(free_vec), sfi,
                det.transform_store());

            // Apply interactor
            typename interactor_type::state interactor_state;
            interactor_type{}.update(
                in_param, interactor_state,
                static_cast<int>(detray::navigation::direction::e_forward), sf,
                sfi.cos_incidence_angle);

            /*************************
             * CKF
             *************************/

            // Get barcode and measurements range on surface
            const auto bcd = in_param.surface_link();
            std::pair<unsigned int, unsigned int> range;

            // Find the corresponding index of bcd in barcode vector
            unsigned int bcd_id;
            if (std::binary_search(barcodes.begin(), barcodes.end(), bcd)) {
                const auto lo2 =
                    std::lower_bound(barcodes.begin(), barcodes.end(), bcd);
                bcd_id = std::distance(barcodes.begin(), lo2);

                if (lo2 == barcodes.begin()) {
                    range.first = 0u;
                    range.second = upper_bounds[bcd_id];
                } else {
                    range.first = upper_bounds[bcd_id - 1];
                    range.second = upper_bounds[bcd_id];
                }
            } else {
                range.first = 0u;
                range.second = 0u;
            }

            unsigned int n_branches = 0;

            // Iterate over the measurements
            for (unsigned int item_id = range.first; item_id < range.second;
                 item_id++) {
                if (n_branches > m_cfg.max_num_branches_per_surface) {
                    break;
                }
                if (n_trks_per_seed[orig_param_id] >=
                    m_cfg.max_num_branches_per_initial_seed) {
                    break;
                }

                bound_track_parameters bound_param(in_param.surface_link(),
                                                   in_param.vector(),
                                                   in_param.covariance());
                const auto& meas = measurements[item_id];

                track_state<transform3_type> trk_state(meas);

                // Run the Kalman update
                sf.template visit_mask<gain_matrix_updater<transform3_type>>(
                    trk_state, bound_param);

                // Get the chi-square
                const auto chi2 = trk_state.filtered_chi2();

                // Found a good measurement
                if (chi2 < m_cfg.chi2_max) {

                    // Current link ID
                    unsigned int cur_link_id =
                        static_cast<unsigned int>(links[step].size());

                    n_branches++;
                    n_trks_per_seed[orig_param_id]++;

                    links[step].push_back({{previous_step, in_param_id},
                                           item_id,
                                           orig_param_id,
                                           skip_counter});

                    /*********************************
                     * Propagate to the next surface
                     *********************************/

                    // Create propagator state
                    typename propagator_type::state propagation(
                        trk_state.filtered(), field, det);
                    propagation._stepping.template set_constraint<
                        detray::step::constraint::e_accuracy>(
                        m_cfg.propagation.stepping.step_constraint);

                    typename detray::pathlimit_aborter::state s0;
                    typename detray::parameter_transporter<
                        transform3_type>::state s1;
                    typename interactor::state s3;
                    typename interaction_register<interactor>::state s2{s3};
                    typename detray::next_surface_aborter::state s4{
                        m_cfg.min_step_length_for_surface_aborter};
                    // typename propagation::print_inspector::state s5{};

                    // @TODO: Should be removed once detray is fixed to set the
                    // volume in the constructor
                    propagation._navigation.set_volume(
                        trk_state.filtered().surface_link().volume());

                    // Propagate to the next surface
                    propagator.propagate_sync(propagation,
                                              std::tie(s0, s1, s2, s3, s4));

                    /*
                    propagator.propagate_sync(propagation,
                                              std::tie(s0, s1, s2, s3, s4, s5));
                    */

                    // If a surface found, add the parameter for the next
                    // step
                    if (s4.success) {
                        out_params.push_back(
                            propagation._stepping._bound_params);
                        param_to_link[step].push_back(cur_link_id);
                    }
                    // Unless the track found a surface, it is considered a
                    // tip
                    else if (!s4.success &&
                             step >= m_cfg.min_track_candidates_per_track - 1) {
                        tips.push_back({step, cur_link_id});
                    }

                    // If no more CKF step is expected, current candidate is
                    // kept as a tip
                    if (s4.success &&
                        step == m_cfg.max_track_candidates_per_track - 1) {
                        tips.push_back({step, cur_link_id});
                    }
                }
            }
            // After the loop over the measurements

            if (n_branches == 0) {
                // let's skip this CKF step for the current track candidate
                if (n_trks_per_seed[orig_param_id] >=
                    m_cfg.max_num_branches_per_initial_seed) {

                    continue;
                }

                bound_track_parameters bound_param(in_param.surface_link(),
                                                   in_param.vector(),
                                                   in_param.covariance());

                measurement dummy_meas;

                dummy_meas.local = in_param.bound_local();
                dummy_meas.variance = dummy_meas.variance + point2{10., 10.};
                dummy_meas.surface_link = in_param.surface_link();

                track_state<transform3_type> trk_state(dummy_meas);

                // Run the Kalman update
                sf.template visit_mask<gain_matrix_updater<transform3_type>>(
                    trk_state, bound_param);

                unsigned int cur_link_id =
                    static_cast<unsigned int>(links[step].size());

                links[step].push_back({{previous_step, in_param_id},
                                       std::numeric_limits<unsigned int>::max(),
                                       orig_param_id,
                                       skip_counter + 1});

                if (skip_counter + 1 > m_cfg.max_num_skipping_per_cand) {
                    tips.push_back({step, cur_link_id});
                    continue;
                    // exit from param_id loop
                }

                // Create propagator state
                typename propagator_type::state propagation(
                    trk_state.filtered(), field, det);
                propagation._stepping.template set_constraint<
                    detray::step::constraint::e_accuracy>(
                    m_cfg.propagation.stepping.step_constraint);

                typename detray::pathlimit_aborter::state s0;
                typename detray::parameter_transporter<transform3_type>::state
                    s1;
                typename interactor::state s3;
                typename interaction_register<interactor>::state s2{s3};
                typename detray::next_surface_aborter::state s4{
                    m_cfg.min_step_length_for_surface_aborter};

                propagation._navigation.set_volume(
                    trk_state.filtered().surface_link().volume());

                // Propagate to the next surface
                propagator.propagate_sync(propagation,
                                          std::tie(s0, s1, s2, s3, s4));

                // If a surface found, add the parameter for the next
                // step
                if (s4.success) {
                    out_params.push_back(propagation._stepping._bound_params);
                    param_to_link[step].push_back(cur_link_id);
                }
                // Unless the track found a surface, it is considered a
                // tip
                else if (!s4.success &&
                         step >= m_cfg.min_track_candidates_per_track - 1) {
                    tips.push_back({step, cur_link_id});
                }
            }
        }

        in_params = std::move(out_params);
        out_params.clear();
    }

    /**********************
     * Build tracks
     **********************/

    // Number of found tracks = number of tips
    output_candidates.reserve(tips.size());

    for (const auto& tip : tips) {

        // Skip if the number of tracks candidates is too small
        if (tip.first + 1 < m_cfg.min_track_candidates_per_track) {
            continue;
        }

        // Get the link corresponding to tip
        auto L = links[tip.first][tip.second];

        // Skip if the number of tracks candidates is too small
        if (tip.first + 1 - L.n_skipped <
            m_cfg.min_track_candidates_per_track) {
            continue;
        }

        vecmem::vector<track_candidate> cands_per_track;
        cands_per_track.resize(tip.first + 1 - L.n_skipped);

        // Reversely iterate to fill the track candidates
        for (auto it = cands_per_track.rbegin(); it != cands_per_track.rend();
             it++) {

            while (L.meas_idx > measurements.size()) {

                if (L.previous.first > tip.first + 1)
                    break;  // should not happen.

                const auto link_pos =
                    param_to_link[L.previous.first][L.previous.second];

                L = links[L.previous.first][link_pos];
            }

            auto& cand = *it;

            cand = measurements.at(L.meas_idx);

            // Break the loop if the iterator is at the first candidate and
            // fill the seed
            if (it == cands_per_track.rend() - 1) {

                auto cand_seed = seeds.at(L.previous.second);

                // Add seed and track candidates to the output container
                output_candidates.push_back(cand_seed, cands_per_track);
                break;
            }

            const auto l_pos =
                param_to_link[L.previous.first][L.previous.second];

            L = links[L.previous.first][l_pos];
        }
    }

    return output_candidates;
}

}  // namespace traccc
