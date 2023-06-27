/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/finding/candidate_link.hpp"
#include "traccc/utils/compare.hpp"

// System include
#include <algorithm>
#include <iostream>
#include <limits>

namespace traccc {

template <typename stepper_t, typename navigator_t>
track_candidate_container_types::host
finding_algorithm<stepper_t, navigator_t>::operator()(
    const detector_type& det,
    const measurement_container_types::host& measurements,
    const bound_track_parameters_collection_types::host& seeds) const {

    track_candidate_container_types::host output_candidates;

    /**********************
     * Create module map
     **********************/

    std::vector<std::pair<geometry_id, unsigned int>> module_map;
    const int n_modules = measurements.size();

    module_map.reserve(n_modules);

    for (int i = 0; i < n_modules; i++) {
        module_map.push_back(
            std::make_pair(measurements.at(i).header.module, i));
    }

    // Sort module map to use it for binary search
    std::sort(module_map.begin(), module_map.end());

    /**********************
     * Find tracks
     **********************/

    std::vector<std::vector<candidate_link>> links;
    links.resize(m_cfg.max_track_candidates_per_track);

    std::vector<std::vector<std::size_t>> param_to_link;
    param_to_link.resize(m_cfg.max_track_candidates_per_track);

    std::vector<typename candidate_link::link_index_type> tips;

    // Create propagator
    propagator_type propagator({}, {});

    // Copy seed to input parameters
    std::vector<bound_track_parameters> in_params;
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

        for (unsigned int in_param_id = 0; in_param_id < n_in_params;
             in_param_id++) {

            bound_track_parameters& in_param = in_params[in_param_id];

            /*************************
             * Material interaction
             *************************/

            // Get intersection at surface
            const auto free_vec = det.bound_to_free_vector(
                in_param.surface_link(), in_param.vector());
            const auto& mask_store = det.mask_store();
            intersection_type sfi;

            const auto sf = det.surfaces(in_param.surface_link());

            sfi.surface = sf;
            mask_store.template visit<detray::intersection_update>(
                sfi.surface.mask(),
                detray::detail::ray<transform3_type>(free_vec), sfi,
                det.transform_store());

            // Apply interactor
            typename interactor_type::state interactor_state;
            interactor_type{}.update(
                in_param, interactor_state,
                static_cast<int>(detray::navigation::direction::e_forward), sfi,
                det.material_store());

            /*************************
             * CKF
             *************************/

            // Get module id
            const auto module_id = in_param.surface_link();

            unsigned int header_id;
            if (std::binary_search(
                    module_map.begin(), module_map.end(), module_id.value(),
                    compare_pair_int<std::pair, unsigned int>())) {
                const auto lo2 = std::lower_bound(
                    module_map.begin(), module_map.end(), module_id.value(),
                    compare_pair_int<std::pair, unsigned int>());
                header_id = (*lo2).second;
            } else {
                continue;
            }

            // Get measurements on surface
            const auto measurements_on_surface =
                measurements.at(header_id).items;

            const unsigned int n_meas = measurements_on_surface.size();
            unsigned int n_branches = 0;

            // Iterate over the measurements
            for (unsigned int item_id = 0; item_id < n_meas; item_id++) {
                if (n_branches > m_cfg.max_num_branches_per_surface) {
                    break;
                }

                // bound_track_parameters bound_param = in_param;
                bound_track_parameters bound_param(in_param.surface_link(),
                                                   in_param.vector(),
                                                   in_param.covariance());
                const auto& meas = measurements_on_surface[item_id];

                track_state<transform3_type> trk_state({module_id, meas});

                // Run the Kalman update
                mask_store.template visit<gain_matrix_updater<transform3_type>>(
                    sf.mask(), trk_state, bound_param);

                // Get the chi-square
                const auto chi2 = trk_state.filtered_chi2();

                // Found a good measurement
                if (chi2 < m_cfg.chi2_max) {

                    // Current link ID
                    unsigned int cur_link_id =
                        static_cast<unsigned int>(links[step].size());

                    n_branches++;

                    links[step].push_back({{previous_step, in_param_id},
                                           {header_id, item_id},
                                           module_id});

                    /*********************************
                     * Propagate to the next surface
                     *********************************/

                    // Create propagator state
                    typename propagator_type::state propagation(
                        trk_state.filtered(), det.get_bfield(), det);
                    propagation._stepping.template set_constraint<
                        detray::step::constraint::e_accuracy>(
                        m_cfg.constrained_step_size);

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

        vecmem::vector<track_candidate> cands_per_track;
        cands_per_track.resize(tip.first + 1);

        // Get the link corresponding to tip
        auto L = links[tip.first][tip.second];

        // Reversely iterate to fill the track candidates
        for (auto it = cands_per_track.rbegin(); it != cands_per_track.rend();
             it++) {

            auto& cand = *it;

            cand = {L.surface_link, measurements.at(L.meas_link)};

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
