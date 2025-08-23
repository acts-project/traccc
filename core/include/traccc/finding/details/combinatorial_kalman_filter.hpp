/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate_collection.hpp"
#include "traccc/edm/track_state_helpers.hpp"
#include "traccc/finding/actors/ckf_aborter.hpp"
#include "traccc/finding/actors/interaction_register.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/details/combinatorial_kalman_filter_types.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"
#include "traccc/fitting/kalman_filter/is_line_visitor.hpp"
#include "traccc/fitting/status_codes.hpp"
#include "traccc/sanity/contiguous_on.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/particle.hpp"
#include "traccc/utils/prob.hpp"
#include "traccc/utils/projections.hpp"
#include "traccc/utils/propagation.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

namespace traccc::host::details {

/// Templated implementation of the track finding algorithm.
///
/// Concrete track finding algorithms can use this function with the appropriate
/// specializations, to find tracks on top of a specific detector type, magnetic
/// field type, and track finding configuration.
///
/// @tparam detector_t The (host) detector type to use
/// @tparam bfield_t   The magnetic field type to use
///
/// @param det               The detector object
/// @param field             The magnetic field object
/// @param measurements_view All measurements in an event
/// @param seeds_view        All seeds in an event to start the track finding
///                          with
/// @param config            The track finding configuration
/// @param mr                The memory resource to use
/// @param log               The logger object to use
///
/// @return A container of the found track candidates
///
template <typename detector_t, typename bfield_t>
edm::track_candidate_collection<default_algebra>::host
combinatorial_kalman_filter(
    const detector_t& det, const bfield_t& field,
    const measurement_collection_types::const_view& measurements_view,
    const bound_track_parameters_collection_types::const_view& seeds_view,
    const finding_config& config, vecmem::memory_resource& mr,
    const Logger& log) {

    assert(config.min_step_length_for_next_surface >
               math::fabs(config.propagation.navigation.overstep_tolerance) &&
           "Min step length for the next surface should be higher than the "
           "overstep tolerance");
    assert(config.min_track_candidates_per_track >= 1);

    /// The algebra type
    using algebra_type = typename detector_t::algebra_type;
    /// The scalar type
    using scalar_type = detray::dscalar<algebra_type>;

    // Create a logger.
    auto logger = [&log]() -> const Logger& { return log; };

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

    std::vector<std::pair<unsigned int, unsigned int>> tips;

    // Create propagator
    traccc::details::ckf_propagator_t<detector_t, bfield_t> propagator(
        config.propagation);

    // Create the input seeds container.
    bound_track_parameters_collection_types::const_device seeds{seeds_view};

    // Copy seed to input parameters
    std::vector<bound_track_parameters<algebra_type>> in_params(seeds.size());
    std::copy(seeds.begin(), seeds.end(), in_params.begin());
    std::vector<unsigned int> n_trks_per_seed(seeds.size());

    std::vector<bound_track_parameters<algebra_type>> out_params;

    for (unsigned int step = 0u; step < config.max_track_candidates_per_track;
         step++) {

        TRACCC_VERBOSE("Starting step "
                       << step + 1 << " / "
                       << config.max_track_candidates_per_track);

        // Iterate over input parameters
        const std::size_t n_in_params = in_params.size();

        // Terminate if there is no parameter to proceed
        if (n_in_params == 0) {
            break;
        }

        // Rough estimation on out parameters size
        out_params.reserve(n_in_params);

        // Previous step ID
        std::fill(n_trks_per_seed.begin(), n_trks_per_seed.end(), 0u);

        // Parameters updated by Kalman fitter
        std::vector<bound_track_parameters<algebra_type>> updated_params;

        for (unsigned int in_param_id = 0; in_param_id < n_in_params;
             in_param_id++) {

            bound_track_parameters<algebra_type>& in_param =
                in_params[in_param_id];

            assert(!in_param.is_invalid());

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
            const scalar prev_chi2_sum =
                (step == 0
                     ? 0.f
                     : links[step - 1][param_to_link[step - 1][in_param_id]]
                           .chi2_sum);
            const unsigned int prev_ndf_sum =
                (step == 0
                     ? 0
                     : links[step - 1][param_to_link[step - 1][in_param_id]]
                           .ndf_sum);

            TRACCC_VERBOSE("Processing input parameter "
                           << in_param_id + 1 << " / " << n_in_params << ": "
                           << in_param << " (orig_param_id=" << orig_param_id
                           << ", skip_counter=" << skip_counter << ")");

            /*************************
             * Material interaction
             *************************/

            // Get surface corresponding to bound params
            const detray::tracking_surface sf{det, in_param.surface_link()};

            TRACCC_VERBOSE(
                "  free params: " << sf.bound_to_free_vector({}, in_param));

            // Apply interactor
            if (sf.has_material()) {
                const typename detector_t::geometry_context ctx{};
                traccc::details::ckf_interactor_t::state interactor_state;
                traccc::details::ckf_interactor_t{}.update(
                    ctx,
                    detail::correct_particle_hypothesis(config.ptc_hypothesis,
                                                        in_param),
                    in_param, interactor_state,
                    static_cast<int>(detray::navigation::direction::e_forward),
                    sf);
            }

            // Get barcode and measurements range on surface
            const auto bcd = in_param.surface_link();
            assert(!bcd.is_invalid());
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

            /*****************************************************************
             * Find tracks (CKF)
             *****************************************************************/

            std::vector<std::tuple<candidate_link,
                                   bound_track_parameters<algebra_type>>>
                best_links;

            // Iterate over the measurements
            for (unsigned int item_id = range.first; item_id < range.second;
                 item_id++) {

                // The measurement on surface to handle.
                const measurement& meas = measurements.at(item_id);

                // Create a standalone track state object.
                auto trk_state =
                    edm::make_track_state<algebra_type>(measurements, item_id);

                const bool is_line = sf.template visit_mask<is_line_visitor>();

                // Run the Kalman update on a copy of the track parameters
                const kalman_fitter_status res =
                    gain_matrix_updater<algebra_type>{}(trk_state, measurements,
                                                        in_param, is_line);

                const traccc::scalar chi2 = trk_state.filtered_chi2();

                // The chi2 from Kalman update should be less than chi2_max
                if (res == kalman_fitter_status::SUCCESS &&
                    chi2 < config.chi2_max) {

                    best_links.push_back(
                        {{.step = step,
                          .previous_candidate_idx = in_param_id,
                          .meas_idx = item_id,
                          .seed_idx = orig_param_id,
                          .n_skipped = skip_counter,
                          .chi2 = chi2,
                          .chi2_sum = prev_chi2_sum + chi2,
                          .ndf_sum = prev_ndf_sum + meas.meas_dim},
                         trk_state.filtered_params()});
                }
            }

            // Sort the links by chi2
            std::sort(best_links.begin(), best_links.end(),
                      [](const auto& a, const auto& b) {
                          return std::get<0>(a).chi2 < std::get<0>(b).chi2;
                      });
            // Take the best links
            const unsigned int n_branches =
                std::min(config.max_num_branches_per_surface,
                         static_cast<unsigned int>(best_links.size()));
            TRACCC_VERBOSE("Found " << n_branches << " branches for step "
                                    << step << " and input parameter "
                                    << in_param_id);
            for (unsigned int i = 0; i < n_branches; ++i) {
                const auto& [link, filtered_params] = best_links[i];

                // Add the link to the links container
                links[step].push_back(link);

                // Add the updated parameter to the updated parameters
                updated_params.push_back(filtered_params);
                TRACCC_VERBOSE("updated_params["
                               << updated_params.size() - 1
                               << "] = " << updated_params.back());
            }

            /*****************************************************************
             * Add a dummy links in case of no branches
             *****************************************************************/

            if (n_branches == 0) {

                // Put an invalid link with max item id
                links[step].push_back(
                    {.step = step,
                     .previous_candidate_idx = in_param_id,
                     .meas_idx = std::numeric_limits<unsigned int>::max(),
                     .seed_idx = orig_param_id,
                     .n_skipped = skip_counter + 1,
                     .chi2 = std::numeric_limits<traccc::scalar>::max(),
                     .chi2_sum = prev_chi2_sum,
                     .ndf_sum = prev_ndf_sum});

                updated_params.push_back(in_param);
                TRACCC_VERBOSE("updated_params["
                               << updated_params.size() - 1
                               << "] = " << updated_params.back());
            }
        }

        /*
         * Track deduplication.
         *
         * For documentation, see the device version.
         */
        const std::size_t n_links = links[step].size();
        std::vector<unsigned int> param_liveness;
        param_liveness.resize(n_links);

        for (std::size_t i = 0; i < param_liveness.size(); ++i) {
            param_liveness.at(i) = 1u;
        }

        if (step >= config.duplicate_removal_minimum_length) {
            std::map<std::size_t, std::vector<std::size_t>>
                last_meas_to_tracks_map;

            for (std::size_t i = 0; i < n_links; ++i) {
                auto L = links.at(step).at(i);

                while (L.meas_idx >= n_meas && L.step != 0u) {
                    const auto link_pos = param_to_link.at(L.step - 1u)
                                              .at(L.previous_candidate_idx);
                    L = links.at(L.step - 1u).at(link_pos);
                }

                last_meas_to_tracks_map[L.meas_idx].push_back(i);
            }

            for (const auto& it : last_meas_to_tracks_map) {
                const auto& tracks = it.second;

                for (std::size_t i = 0; i < tracks.size(); ++i) {
                    const auto& Lthisbase = links.at(step).at(tracks.at(i));
                    const scalar prob_this =
                        prob(Lthisbase.chi2_sum,
                             static_cast<scalar>(Lthisbase.ndf_sum - 5));

                    if (step + 1 - Lthisbase.n_skipped <=
                            config.duplicate_removal_minimum_length ||
                        Lthisbase.ndf_sum <= 5) {
                        continue;
                    }

                    for (std::size_t j = 0; j < tracks.size(); ++j) {
                        if (i == j) {
                            continue;
                        }

                        auto Lthis = Lthisbase;
                        auto Lthat = links.at(step).at(tracks.at(j));

                        if (step + 1 - Lthat.n_skipped <=
                                config.duplicate_removal_minimum_length ||
                            Lthisbase.ndf_sum <= 5) {
                            continue;
                        }

                        bool this_is_dominated = true;

                        const scalar prob_that =
                            prob(Lthat.chi2_sum,
                                 static_cast<scalar>(Lthat.ndf_sum - 5));

                        while (true) {
                            while (Lthis.meas_idx >= n_meas &&
                                   Lthis.step != 0u) {
                                const auto link_pos =
                                    param_to_link.at(Lthis.step - 1u)
                                        .at(Lthis.previous_candidate_idx);

                                Lthis = links.at(Lthis.step - 1u).at(link_pos);
                            }
                            while (Lthat.meas_idx >= n_meas &&
                                   Lthat.step != 0u) {
                                const auto link_pos =
                                    param_to_link.at(Lthat.step - 1u)
                                        .at(Lthat.previous_candidate_idx);

                                Lthat = links.at(Lthat.step - 1u).at(link_pos);
                            }

                            if (Lthis.meas_idx == Lthat.meas_idx) {
                                if (Lthis.step == 0) {
                                    break;
                                } else if (Lthat.step == 0) {
                                    this_is_dominated = false;
                                    break;
                                } else {
                                    const auto link_pos_this =
                                        param_to_link.at(Lthis.step - 1u)
                                            .at(Lthis.previous_candidate_idx);
                                    Lthis = links.at(Lthis.step - 1u)
                                                .at(link_pos_this);
                                    const auto link_pos_that =
                                        param_to_link.at(Lthat.step - 1u)
                                            .at(Lthat.previous_candidate_idx);
                                    Lthat = links.at(Lthat.step - 1u)
                                                .at(link_pos_that);
                                }
                            } else {
                                this_is_dominated = false;
                                break;
                            }
                        }

                        if (prob_this != prob_that) {
                            this_is_dominated &= prob_that >= prob_this;
                        } else {
                            this_is_dominated &= tracks.at(j) < tracks.at(i);
                        }

                        if (this_is_dominated) {
                            param_liveness.at(tracks.at(i)) = 0u;
                            break;
                        }
                    }
                }
            }
        }

        /*********************************
         * Propagate to the next surface
         *********************************/
        for (unsigned int link_id = 0; link_id < n_links; link_id++) {
            if (param_liveness.at(link_id) == 0u) {
                continue;
            }

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
            typename traccc::details::ckf_propagator_t<
                detector_t, bfield_t>::state propagation(param, field, det);
            propagation.set_particle(detail::correct_particle_hypothesis(
                config.ptc_hypothesis, param));

            propagation._stepping
                .template set_constraint<detray::step::constraint::e_accuracy>(
                    config.propagation.stepping.step_constraint);

            typename detray::pathlimit_aborter<scalar_type>::state s0;
            traccc::details::ckf_interactor_t::state s2;
            typename interaction_register<
                traccc::details::ckf_interactor_t>::state s1{s2};
            typename detray::momentum_aborter<scalar_type>::state s3{};
            typename ckf_aborter::state s4;
            // Update the actor config
            s4.min_step_length = config.min_step_length_for_next_surface;
            s4.max_count = config.max_step_counts_for_next_surface;
            s3.min_pT(static_cast<scalar_type>(config.min_pT));
            s3.min_p(static_cast<scalar_type>(config.min_p));

            // Propagate to the next surface
            propagator.propagate(propagation, detray::tie(s0, s1, s2, s3, s4));

            // If a surface found, add the parameter for the next
            // step
            if (s4.success) {
                assert(propagation._navigation.is_on_sensitive());
                assert(!propagation._stepping.bound_params().is_invalid());

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
    typename edm::track_candidate_collection<algebra_type>::host
        output_candidates{mr};
    output_candidates.reserve(tips.size());

    for (const auto& tip : tips) {
        // Get the link corresponding to tip
        auto L = links.at(tip.first).at(tip.second);

        const unsigned int n_cands = tip.first + 1 - L.n_skipped;

        // Skip if the number of tracks candidates is too small
        if (n_cands < config.min_track_candidates_per_track ||
            n_cands > config.max_track_candidates_per_track) {
            continue;
        }

        // Retrieve tip
        L = links.at(tip.first).at(tip.second);

        vecmem::vector<unsigned int> cands_per_track;
        cands_per_track.resize(n_cands);

        // Track summary variables
        scalar ndf_sum = 0.f;
        scalar chi2_sum = 0.f;

        // Reversely iterate to fill the track candidates
        for (auto it = cands_per_track.rbegin(); it != cands_per_track.rend();
             it++) {

            while (L.meas_idx >= n_meas && L.step != 0u) {
                const auto link_pos =
                    param_to_link.at(L.step - 1u).at(L.previous_candidate_idx);

                L = links.at(L.step - 1u).at(link_pos);
            }

            // Break if the measurement is still invalid
            if (L.meas_idx >= measurements.size()) {
                break;
            }

            *it = L.meas_idx;

            // Sanity check on chi2
            assert(L.chi2 < std::numeric_limits<traccc::scalar>::max());
            assert(L.chi2 >= 0.f);

            ndf_sum += static_cast<scalar>(measurements.at(*it).meas_dim);
            chi2_sum += L.chi2;

            // Break the loop if the iterator is at the first candidate and
            // fill the seed
            if (it == cands_per_track.rend() - 1) {

                auto cand_seed = seeds.at(L.seed_idx);
                ndf_sum = ndf_sum - 5.f;
                const auto pval = prob(chi2_sum, ndf_sum);

                // Add seed and track candidates to the output container
                output_candidates.push_back({cand_seed, ndf_sum, chi2_sum, pval,
                                             L.n_skipped, cands_per_track});
            } else {
                const auto l_pos =
                    param_to_link.at(L.step - 1u).at(L.previous_candidate_idx);

                L = links.at(L.step - 1u).at(l_pos);
            }
        }
    }

    return output_candidates;
}

}  // namespace traccc::host::details
