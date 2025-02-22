/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/finding/actors/ckf_aborter.hpp"
#include "traccc/finding/actors/interaction_register.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"
#include "traccc/fitting/status_codes.hpp"
#include "traccc/sanity/contiguous_on.hpp"
#include "traccc/utils/particle.hpp"
#include "traccc/utils/projections.hpp"

// Detray include(s).
#include <detray/propagator/actors.hpp>
#include <detray/propagator/propagator.hpp>

// System include(s).
#include <algorithm>
#include <cassert>
#include <vector>

namespace traccc::host::details {

/// Templated implementation of the track finding algorithm.
///
/// Concrete track finding algorithms can use this function with the appropriate
/// specializations, to find tracks on top of a specific detector type, magnetic
/// field type, and track finding configuration.
///
/// @tparam stepper_t The stepper type used for the track propagation
/// @tparam navigator_t The navigator type used for the track navigation
///
/// @param det               The detector object
/// @param field             The magnetic field object
/// @param measurements_view All measurements in an event
/// @param seeds_view        All seeds in an event to start the track finding
///                          with
/// @param config            The track finding configuration
///
/// @return A container of the found track candidates
///
template <typename stepper_t, typename navigator_t>
track_candidate_container_types::host find_tracks(
    const typename navigator_t::detector_type& det,
    const typename stepper_t::magnetic_field_type& field,
    const measurement_collection_types::const_view& measurements_view,
    const bound_track_parameters_collection_types::const_view& seeds_view,
    const finding_config& config) {

    /*****************************************************************
     * Types used by the track finding
     *****************************************************************/

    using algebra_type = typename navigator_t::detector_type::algebra_type;
    using scalar_type = detray::dscalar<algebra_type>;

    using transporter_type = detray::parameter_transporter<algebra_type>;
    using interactor_type = detray::pointwise_material_interactor<algebra_type>;

    using actor_type = detray::actor_chain<
        detray::pathlimit_aborter<scalar_type>, transporter_type,
        interaction_register<interactor_type>, interactor_type, ckf_aborter>;

    using propagator_type =
        detray::propagator<stepper_t, navigator_t, actor_type>;

    assert(config.min_track_candidates_per_track >= 1);

    /*****************************************************************
     * Measurement Operations
     *****************************************************************/

    // Create the measurement container.
    measurement_collection_types::const_device measurements{measurements_view};

    // Check contiguity of the measurements
    assert(is_contiguous_on(measurement_module_projection(), measurements));

    // Get copy of barcode uniques
    std::vector<measurement> uniques;
    uniques.resize(measurements.size());

    std::vector<measurement>::iterator uniques_end =
        std::unique_copy(measurements.begin(), measurements.end(),
                         uniques.begin(), measurement_equal_comp());
    const auto n_modules =
        static_cast<unsigned int>(uniques_end - uniques.begin());

    // Get upper bounds of unique elements
    std::vector<unsigned int> upper_bounds;
    upper_bounds.reserve(n_modules);
    for (unsigned int i = 0; i < n_modules; i++) {
        measurement_collection_types::const_device::iterator up =
            std::upper_bound(measurements.begin(), measurements.end(),
                             uniques[i], measurement_sort_comp());
        upper_bounds.push_back(
            static_cast<unsigned int>(std::distance(measurements.begin(), up)));
    }
    const measurement_collection_types::const_device::size_type n_meas =
        measurements.size();

    // Get the number of measurements of each module
    std::vector<unsigned int> sizes(n_modules);
    std::adjacent_difference(upper_bounds.begin(), upper_bounds.end(),
                             sizes.begin());

    // Create barcode sequence
    std::vector<detray::geometry::barcode> barcodes(n_modules);
    std::transform(uniques.begin(), uniques_end, barcodes.begin(),
                   [](const measurement& m) { return m.surface_link; });

    std::vector<std::vector<candidate_link>> links;
    links.resize(config.max_track_candidates_per_track);

    std::vector<std::vector<std::size_t>> param_to_link;
    param_to_link.resize(config.max_track_candidates_per_track);

    std::vector<typename candidate_link::link_index_type> tips;

    // Create propagator
    propagator_type propagator(config.propagation);

    // Create the input seeds container.
    bound_track_parameters_collection_types::const_device seeds{seeds_view};

    // Copy seed to input parameters
    std::vector<bound_track_parameters<algebra_type>> in_params(seeds.size());
    std::copy(seeds.begin(), seeds.end(), in_params.begin());
    std::vector<unsigned int> n_trks_per_seed(seeds.size());

    std::vector<bound_track_parameters<algebra_type>> out_params;

    for (unsigned int step = 0u; step < config.max_track_candidates_per_track;
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
        const candidate_link::link_index_type::first_type previous_step =
            (step == 0u)
                ? std::numeric_limits<
                      candidate_link::link_index_type::first_type>::max()
                : step - 1u;

        std::fill(n_trks_per_seed.begin(), n_trks_per_seed.end(), 0u);

        // Parameters updated by Kalman fitter
        std::vector<bound_track_parameters<algebra_type>> updated_params;

        for (unsigned int in_param_id = 0; in_param_id < n_in_params;
             in_param_id++) {

            bound_track_parameters<algebra_type>& in_param =
                in_params[in_param_id];
            const unsigned int orig_param_id =
                (step == 0
                     ? in_param_id
                     : links[step - 1][param_to_link[step - 1][in_param_id]]
                           .seed_idx);
            const unsigned int skip_counter =
                (step == 0
                     ? 0
                     : links[step - 1][param_to_link[step - 1][in_param_id]]
                           .n_skipped);

            /*************************
             * Material interaction
             *************************/

            // Get surface corresponding to bound params
            const detray::tracking_surface sf{det, in_param.surface_link()};

            const typename navigator_t::detector_type::geometry_context ctx{};

            // Apply interactor
            typename interactor_type::state interactor_state;
            interactor_type{}.update(
                ctx,
                detail::correct_particle_hypothesis(config.ptc_hypothesis,
                                                    in_param),
                in_param, interactor_state,
                static_cast<int>(detray::navigation::direction::e_forward), sf);

            // Get barcode and measurements range on surface
            const auto bcd = in_param.surface_link();
            std::pair<unsigned int, unsigned int> range;

            // Find the corresponding index of bcd in barcode vector

            const auto lo2 =
                std::lower_bound(barcodes.begin(), barcodes.end(), bcd);

            const auto bcd_id = std::distance(barcodes.begin(), lo2);

            if (lo2 == barcodes.begin()) {
                range.first = 0u;
                range.second = upper_bounds[static_cast<std::size_t>(bcd_id)];
            } else if (lo2 == barcodes.end()) {
                range.first = 0u;
                range.second = 0u;
            } else {
                range.first =
                    upper_bounds[static_cast<std::size_t>(bcd_id - 1)];
                range.second = upper_bounds[static_cast<std::size_t>(bcd_id)];
            }

            unsigned int n_branches = 0;

            /*****************************************************************
             * Find tracks (CKF)
             *****************************************************************/

            // Iterate over the measurements
            for (unsigned int item_id = range.first; item_id < range.second;
                 item_id++) {
                if (n_branches > config.max_num_branches_per_surface) {
                    break;
                }

                const auto& meas = measurements[item_id];

                track_state<algebra_type> trk_state(meas);

                // Run the Kalman update on a copy of the track parameters
                const kalman_fitter_status res =
                    sf.template visit_mask<gain_matrix_updater<algebra_type>>(
                        trk_state, in_param);

                const traccc::scalar chi2 = trk_state.filtered_chi2();

                // The chi2 from Kalman update should be less than chi2_max
                if (res == kalman_fitter_status::SUCCESS &&
                    chi2 < config.chi2_max) {
                    n_branches++;

                    links[step].push_back({{previous_step, in_param_id},
                                           item_id,
                                           orig_param_id,
                                           skip_counter,
                                           chi2});
                    updated_params.push_back(trk_state.filtered());
                }
            }

            /*****************************************************************
             * Add a dummy links in case of no branches
             *****************************************************************/

            if (n_branches == 0) {

                // Put an invalid link with max item id
                links[step].push_back(
                    {{previous_step, in_param_id},
                     std::numeric_limits<unsigned int>::max(),
                     orig_param_id,
                     skip_counter + 1,
                     std::numeric_limits<traccc::scalar>::max()});

                updated_params.push_back(in_param);
                n_branches++;
            }
        }

        /*********************************
         * Propagate to the next surface
         *********************************/

        const std::size_t n_links = links[step].size();
        for (unsigned int link_id = 0; link_id < n_links; link_id++) {

            const unsigned int seed_idx = links.at(step).at(link_id).seed_idx;
            n_trks_per_seed[seed_idx]++;

            if (n_trks_per_seed[seed_idx] > config.max_num_branches_per_seed) {
                continue;
            }

            // If number of skips is larger than the maximum value, consider the
            // link to be a tip
            if (links.at(step).at(link_id).n_skipped >
                config.max_num_skipping_per_cand) {
                tips.push_back({step, link_id});
                continue;
            }

            const auto& param = updated_params[link_id];
            // Create propagator state
            typename propagator_type::state propagation(param, field, det);
            propagation.set_particle(detail::correct_particle_hypothesis(
                config.ptc_hypothesis, param));

            propagation._stepping
                .template set_constraint<detray::step::constraint::e_accuracy>(
                    config.propagation.stepping.step_constraint);

            typename detray::pathlimit_aborter<scalar_type>::state s0;
            typename detray::parameter_transporter<algebra_type>::state s1;
            typename interactor_type::state s3;
            typename interaction_register<interactor_type>::state s2{s3};
            typename ckf_aborter::state s4;
            s4.min_step_length = config.min_step_length_for_next_surface;
            s4.max_count = config.max_step_counts_for_next_surface;

            // @TODO: Should be removed once detray is fixed to set the
            // volume in the constructor
            propagation._navigation.set_volume(param.surface_link().volume());

            // Propagate to the next surface
            propagator.propagate_sync(propagation,
                                      detray::tie(s0, s1, s2, s3, s4));

            // If a surface found, add the parameter for the next
            // step
            if (s4.success) {
                out_params.push_back(propagation._stepping.bound_params());
                param_to_link[step].push_back(link_id);
            }
            // Unless the track found a surface, it is considered a
            // tip
            else if (!s4.success &&
                     (step >= (config.min_track_candidates_per_track - 1u))) {
                tips.push_back({step, link_id});
            }

            // If no more CKF step is expected, current candidate is
            // kept as a tip
            if (s4.success &&
                (step == (config.max_track_candidates_per_track - 1u))) {
                tips.push_back({step, link_id});
            }
        }

        in_params = std::move(out_params);
        out_params.clear();
    }

    /**********************
     * Build tracks
     **********************/

    // Number of found tracks = number of tips
    track_candidate_container_types::host output_candidates;
    output_candidates.reserve(tips.size());

    for (const auto& tip : tips) {
        // Get the link corresponding to tip
        auto L = links.at(tip.first).at(tip.second);

        // Count the number of skipped steps
        unsigned int n_skipped{0u};
        while (true) {

            if (L.meas_idx >= n_meas) {
                n_skipped++;
            }

            if (L.previous.first == 0u) {
                break;
            }

            const unsigned long link_pos =
                param_to_link.at(L.previous.first).at(L.previous.second);
            L = links.at(L.previous.first).at(link_pos);
        }

        const unsigned int n_cands = tip.first + 1 - n_skipped;

        // Skip if the number of tracks candidates is too small
        if (n_cands < config.min_track_candidates_per_track ||
            n_cands > config.max_track_candidates_per_track) {
            continue;
        }

        // Retrieve tip
        L = links.at(tip.first).at(tip.second);

        vecmem::vector<track_candidate> cands_per_track;
        cands_per_track.resize(n_cands);

        // Track summary variables
        scalar ndf_sum = 0.f;
        scalar chi2_sum = 0.f;

        // Reversely iterate to fill the track candidates
        for (auto it = cands_per_track.rbegin(); it != cands_per_track.rend();
             it++) {

            while (
                L.meas_idx >= n_meas &&
                L.previous.first !=
                    std::numeric_limits<
                        candidate_link::link_index_type::first_type>::max()) {
                const auto link_pos =
                    param_to_link.at(L.previous.first).at(L.previous.second);

                L = links.at(L.previous.first).at(link_pos);
            }

            // Break if the measurement is still invalid
            if (L.meas_idx >= measurements.size()) {
                break;
            }

            *it = measurements.at(L.meas_idx);

            // Sanity check on chi2
            assert(L.chi2 < std::numeric_limits<traccc::scalar>::max());
            assert(L.chi2 >= 0.f);

            ndf_sum += static_cast<scalar>(it->meas_dim);
            chi2_sum += L.chi2;

            // Break the loop if the iterator is at the first candidate and
            // fill the seed
            if (it == cands_per_track.rend() - 1) {

                auto cand_seed = seeds.at(L.previous.second);

                // Add seed and track candidates to the output container
                output_candidates.push_back(
                    finding_result{
                        cand_seed,
                        track_quality{ndf_sum - 5.f, chi2_sum, L.n_skipped}},
                    cands_per_track);
            } else {
                const auto l_pos =
                    param_to_link.at(L.previous.first).at(L.previous.second);

                L = links.at(L.previous.first).at(l_pos);
            }
        }
    }

    return output_candidates;
}

}  // namespace traccc::host::details
