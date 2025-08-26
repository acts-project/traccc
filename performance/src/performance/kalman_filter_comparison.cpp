/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/bfield/construct_const_bfield.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/fitting/kalman_filter/kalman_actor.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/data_format.hpp"
#include "traccc/utils/event_data.hpp"
#include "traccc/utils/propagation.hpp"

// Performance include(s).
#include "traccc/performance/kalman_filter_comparison.hpp"
#include "traccc/utils/transcribe_to_trace.hpp"

// Detray test include(s)
#include <detray/test/utils/perigee_stopper.hpp>
#include <detray/test/validation/navigation_validation_utils.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <exception>
#include <iostream>
#include <memory>
#include <string>

namespace traccc {

bool kalman_filter_comparison(
    const traccc::default_detector::host& det,
    const traccc::default_detector::host::name_map& names,
    const detray::propagation::config& prop_cfg, const std::string& input_dir,
    const unsigned int n_events, std::unique_ptr<const traccc::Logger> ilogger,
    const bool do_multiple_scattering, const bool do_energy_loss,
    const bool use_acts_geoid,
    const traccc::pdg_particle<traccc::scalar> ptc_type,
    const std::array<traccc::scalar, traccc::e_bound_size>& stddevs,
    const traccc::vector3& B, const traccc::scalar min_pT,
    const traccc::scalar max_rad) {

    using namespace traccc;

    TRACCC_LOCAL_LOGGER(std::move(ilogger));
    // 'false' if any failures were detected
    bool test_successful{true};

    using detector_t = traccc::default_detector::host;
    using algebra_t = typename detector_t::algebra_type;
    using scalar_t = detray::dscalar<algebra_t>;
    using sf_candidate_t =
        traccc::propagation_validator::candidate_type<detector_t>;
    using b_field_t =
        covfie::field<traccc::const_bfield_backend_t<traccc::scalar>>;
    using stepper_t =
        detray::rk_stepper<b_field_t::view_t, algebra_t,
                           detray::constrained_step<traccc::scalar>,
                           detray::stepper_rk_policy<traccc::scalar>,
                           detray::stepping::print_inspector>;

    // Host memory resource
    vecmem::host_memory_resource host_mr;

    // Geometry context
    const detector_t::geometry_context ctx{};

    // Create B field
    b_field_t field = traccc::construct_const_bfield(B)
                          .as_field<traccc::const_bfield_backend_t<scalar_t>>();
    b_field_t::view_t field_view = field;

    // Collect data for comparison

    // Initial track parameters from truth particle
    std::vector<traccc::free_track_parameters<algebra_t>> tracks{};
    // The traces of truth hits forward and in reverse order
    std::vector<vecmem::vector<sf_candidate_t>> truth_traces_fw{};
    std::vector<vecmem::vector<sf_candidate_t>> truth_traces_bw{};
    // Track states containing truth measurements for the KF
    std::vector<vecmem::vector<track_state<algebra_t>>> track_state_coll{};

    tracks.reserve(n_events * 1000);
    truth_traces_fw.reserve(tracks.capacity());
    truth_traces_bw.reserve(tracks.capacity());

    // Read the truth data in
    for (std::size_t i_event = 0u; i_event < n_events; ++i_event) {
        traccc::event_data evt_data(input_dir, i_event, host_mr, use_acts_geoid,
                                    &det, data_format::csv);

        assert(!evt_data.m_particle_map.empty());
        assert(!evt_data.m_ptc_to_meas_map.empty());

        for (const auto& [ptc_id, ptc] : evt_data.m_particle_map) {
            if (!evt_data.m_ptc_to_meas_map.contains(ptc)) {
                continue;
            }
            // Minimum momentum
            const traccc::scalar pT{vector::perp(ptc.momentum)};
            if (pT <= min_pT) {
                TRACCC_INFO("Removing particle "
                            << ptc_id << " due to transv. momentum cut (pT was "
                            << pT / traccc::unit<traccc::scalar>::MeV
                            << " MeV)");
                continue;
            }

            // Make a trace of detray-understandable intersections
            auto truth_trace_fw =
                traccc::propagation_validator::transcribe_to_trace(
                    ctx, det, ptc, evt_data.m_ptc_to_meas_map);

            if (!truth_trace_fw.empty()) {
                // Minimum radius (remove secondaries)
                const traccc::scalar rad{
                    vector::perp(truth_trace_fw.front().pos)};
                if (rad >= max_rad) {
                    TRACCC_INFO("Removing particle "
                                << ptc_id << " due to radius cut (radius was "
                                << rad / traccc::unit<traccc::scalar>::mm
                                << " mm)");
                    continue;
                }

                // Revert the forward trace for the backward propagation
                vecmem::vector<sf_candidate_t> truth_trace_bw(
                    truth_trace_fw.size());
                std::ranges::reverse_copy(truth_trace_fw,
                                          truth_trace_bw.begin());

                assert(!truth_trace_bw.empty());

                // Construct initial track parameters
                // @TODO: Need volume grid in case of large vertex smearing
                tracks.emplace_back(ptc.vertex, 0.f, ptc.momentum, ptc.charge);

                truth_traces_fw.push_back(std::move(truth_trace_fw));
                truth_traces_bw.push_back(std::move(truth_trace_bw));

                // Transcribe measurements to track states for the KF
                const auto& measurements = evt_data.m_ptc_to_meas_map.at(ptc);
                assert(!measurements.empty());

                vecmem::vector<track_state<algebra_t>> track_states{};
                track_states.reserve(measurements.size());

                for (const auto& meas : measurements) {
                    track_states.emplace_back(meas);
                }

                track_state_coll.push_back(std::move(track_states));
            }
        }
    }

    // Check truth data
    if (truth_traces_fw.empty() || truth_traces_bw.empty()) {
        TRACCC_ERROR("Propagation truth data empty");
        return false;
    }
    if (track_state_coll.empty()) {
        TRACCC_ERROR("Kalman Filter truth data empty");
        return false;
    }

    using perigee_stopper = detray::perigee_stopper<algebra_t>;
    using transporter = detray::parameter_transporter<algebra_t>;
    using interactor = detray::pointwise_material_interactor<algebra_t>;
    using resetter = detray::parameter_resetter<algebra_t>;

    // Run the navigation and compare
    detray::test::navigation_validation_config<algebra_t> test_cfg{};
    test_cfg.n_tracks(tracks.size()).ptc_hypothesis(ptc_type);
    test_cfg.collect_sensitives_only(true).fail_on_diff(false);
    test_cfg.display_only_missed(true).verbose(false);

    // Make a tuple of references from a tuple
    auto setup_actor_states = []<typename... T>(detray::dtuple<T...> & t) {
        return detray::tie(detray::detail::get<T>(t)...);
    };

    // Safe the original truth traces before dummy records are inserted for
    // missing intersections
    auto truth_traces_fw_KF = truth_traces_fw;
    auto truth_traces_bw_KF = truth_traces_bw;

    // Initial state uncertainty
    vecmem::vector<std::array<scalar, e_bound_size>> stddevs_per_track{};

    // Prepare the fitter state for every track
    for (std::size_t i = 0u; i < tracks.size(); ++i) {
        stddevs_per_track.push_back(stddevs);
    }

    // Reusable actor states
    perigee_stopper::state stopper_state{};
    resetter::state resetter_state{};
    resetter_state.n_stddev = prop_cfg.navigation.n_scattering_stddev;
    resetter_state.accumulated_error = prop_cfg.navigation.accumulated_error;
    resetter_state.estimate_scattering_noise = prop_cfg.navigation.estimate_scattering_noise;
    interactor::state interactor_state{};
    interactor_state.do_multiple_scattering = do_multiple_scattering;
    interactor_state.do_energy_loss = do_energy_loss;

    {
        /*std::cout << "-----------------------------------"
                  << "\nFORWARD - No KF" << std::endl
                  << "-----------------------------------\n";

        constexpr double rel_mat_error{0.01};

        // Prepare actor states
        auto state_tuple = detray::make_tuple(interactor_state, resetter_state);
        auto state_ref_tuple = setup_actor_states(state_tuple);
        auto state_ref_tuples =
            vecmem::vector<decltype(state_ref_tuple)>{state_ref_tuple};

        // Forward navigation
        test_cfg.name(det.name(names) + "_GeV_fw");
        test_cfg.navigation_direction(detray::navigation::direction::e_forward);
        const auto [trk_stats_fw, n_surfaces_fw, n_miss_nav_fw, n_miss_truth_fw,
                    step_traces_fw, mat_traces_fw, mat_records_fw] =
            detray::navigation_validator::compare_to_navigation<
                stepper_t, transporter, interactor, resetter>(
                test_cfg, host_mr, det, names, ctx, field_view, prop_cfg,
                truth_traces_fw, tracks, state_ref_tuples, stddevs_per_track);

        std::cout << "BACKWARD - No KF" << std::endl
                  << "-----------------------------------\n";
        auto bw_state_tuple =
            detray::make_tuple(interactor_state, resetter_state, stopper_state);
        auto bw_state_ref_tuple = setup_actor_states(bw_state_tuple);
        auto bw_state_ref_tuples =
            vecmem::vector<decltype(bw_state_ref_tuple)>{bw_state_ref_tuple};

        // Backward navigation
        test_cfg.name(det.name(names) + "_GeV_bw");
        test_cfg.navigation_direction(
            detray::navigation::direction::e_backward);
        const auto [trk_stats_bw, n_surfaces_bw, n_miss_nav_bw, n_miss_truth_bw,
                    step_traces_bw, mat_traces_bw, mat_records_bw] =
            detray::navigation_validator::compare_to_navigation<
                stepper_t, transporter, interactor, resetter, perigee_stopper>(
                test_cfg, host_mr, det, names, ctx, field_view, prop_cfg,
                truth_traces_bw, tracks, bw_state_ref_tuples,
                stddevs_per_track);

        // Make sure some data was collected
        assert(trk_stats_fw.n_tracks > 0u);
        assert(n_surfaces_fw.n_total() > 0u);
        assert(trk_stats_bw.n_tracks > 0u);
        assert(n_surfaces_bw.n_total() > 0u);

        assert(trk_stats_fw.n_tracks == trk_stats_bw.n_tracks);

        // Check, the amount of collected material between forward and backward
        assert(step_traces_fw.size() == trk_stats_fw.n_tracks);
        assert(mat_traces_fw.size() == trk_stats_fw.n_tracks);
        assert(mat_records_fw.size() == trk_stats_fw.n_tracks);
        assert(mat_records_fw.size() == mat_records_bw.size());
        assert(step_traces_fw.size() == step_traces_bw.size());
        assert(mat_traces_fw.size() == mat_traces_bw.size());

        std::cout << "MATERIAL TRACE - No KF" << std::endl
                  << "-----------------------------------\n";

        // Material traces contain different surfaces
        std::size_t n_bad_comp{0u};
        // Overall integrated material differs while surface seq. is identical
        std::size_t n_diff_mat{0u};

        // Loop over tracks
        for (std::size_t i = 0u; i < mat_records_fw.size(); ++i) {
            // No material on that track (e.g. detector model without material)
            if (mat_traces_fw[i].empty() && mat_traces_bw[i].empty()) {
                continue;
            }

            std::remove_cvref_t<decltype(mat_traces_bw[i])> inv_mat_trace_bw{};
            if (!mat_traces_bw[i].empty()) {
                inv_mat_trace_bw.resize(mat_traces_bw[i].size());

                // Revert the backward trace to comapre to the forward trace
                std::ranges::reverse_copy(mat_traces_bw[i],
                                          inv_mat_trace_bw.begin());

                assert(mat_traces_bw[i].size() == inv_mat_trace_bw.size());
                assert(mat_traces_bw[i].front().bcd ==
                       inv_mat_trace_bw.back().bcd);
            }

            // Compare the material traces and total integrated material per trk
            const auto [is_bad_comp, is_diff_mat] =
                detray::material_validator::compare_traces(
                    mat_traces_fw[i], mat_records_fw[i], inv_mat_trace_bw,
                    mat_records_bw[i], i, rel_mat_error, test_cfg.verbose());

            if (is_bad_comp) {
                n_bad_comp++;
            }
            if (is_diff_mat) {
                n_diff_mat++;
            }
        }

        auto n_tracks{static_cast<double>(trk_stats_fw.n_tracks)};

        std::cout << "Total no. tracks with diff. material: "
                  << (n_bad_comp + n_diff_mat) << " ("
                  << 100. * static_cast<double>(n_bad_comp + n_diff_mat) /
                         n_tracks
                  << "%)" << std::endl;

        std::cout << "No. identical tracks with diff. material: " << n_diff_mat
                  << " (" << 100. * static_cast<double>(n_diff_mat) / n_tracks
                  << "%)" << std::endl;
        std::cout << "-----------------------------------\n" << std::endl;

        // Trigger test failures
        if (n_diff_mat != 0) {
            TRACCC_ERROR("" << n_diff_mat << " tracks have differing material");
            test_successful = false;
        }*/
        /*

        // Check stats
        EXPECT_TRUE(static_cast<double>(trk_stats_fw.n_tracks_w_holes) /
                        n_tracks <=
                    std::get<2>(GetParam()));
        EXPECT_TRUE(static_cast<double>(trk_stats_fw.n_tracks_w_extra) /
                        n_tracks <=
                    std::get<3>(GetParam()));
        EXPECT_TRUE(trk_stats_fw.n_max_missed_per_trk <=
                    std::get<4>(GetParam()));

        n_tracks = static_cast<double>(trk_stats_bw.n_tracks);
        EXPECT_TRUE(static_cast<double>(trk_stats_bw.n_tracks_w_holes) /
                        n_tracks <=
                    std::get<2>(GetParam()));
        EXPECT_TRUE(static_cast<double>(trk_stats_bw.n_tracks_w_extra) /
                        n_tracks <=
                    std::get<3>(GetParam()));
        EXPECT_TRUE(trk_stats_bw.n_max_missed_per_trk <=
                    std::get<4>(GetParam()));*/
    }

    {
        std::cout << "-----------------------------------"
                  << "\nFORWARD - With KF" << std::endl
                  << "-----------------------------------\n";

        using fit_actor_fw =
            traccc::kalman_actor<algebra_t,
                                 kalman_actor_direction::FORWARD_ONLY>;
        using actor_chain_t = detray::actor_chain<transporter, interactor,
                                                  fit_actor_fw, resetter>;

        // Safe track states before KF modifies them
        auto track_states_coll_bw = track_state_coll;
        vecmem::vector<typename actor_chain_t::state_tuple> state_tuple{};
        vecmem::vector<typename actor_chain_t::state_ref_tuple>
            state_ref_tuple{};
        state_tuple.reserve(tracks.size());
        state_ref_tuple.reserve(tracks.size());

        // Prepare the fitter state for every track
        for (std::size_t i = 0u; i < tracks.size(); ++i) {
            auto& track_states = track_state_coll[i];
            // Prepare actor states
            auto trk_states_view = vecmem::get_data(track_states);
            fit_actor_fw::state fit_actor_state{
                vecmem::device_vector<track_state<algebra_t>>(trk_states_view)};
            fit_actor_state.do_precise_hole_count = true;

            state_tuple.push_back(
                detray::make_tuple(interactor_state, fit_actor_state, resetter_state));
            state_ref_tuple.push_back(setup_actor_states(state_tuple.back()));
        }

        // Forward filter
        test_cfg.name(det.name(names) + "_GeV_fw_KF");
        test_cfg.navigation_direction(detray::navigation::direction::e_forward);
        const auto [trk_stats_fw, n_surfaces_fw, n_miss_nav_fw, n_miss_truth_fw,
                    step_traces_fw, mat_traces_fw, mat_records_fw] =
            detray::navigation_validator::compare_to_navigation<
                stepper_t, transporter, interactor, fit_actor_fw, resetter>(
                test_cfg, host_mr, det, names, ctx, field_view, prop_cfg,
                truth_traces_fw_KF, tracks, state_ref_tuple, stddevs_per_track);

        // Check, how many holes the KF found
        std::size_t n_missed_fw{0u};
        std::size_t n_trk_missing_fw{0u};
        std::size_t n_holes_fw{0u};
        std::size_t n_trk_holes_fw{0u};
        for (std::size_t i = 0u; i < tracks.size(); ++i) {
            const auto& actor_states = state_tuple[i];
            auto fitter_state = detray::get<fit_actor_fw::state>(actor_states);

            // What the actor counted
            n_missed_fw += fitter_state.count_missed();
            n_holes_fw += fitter_state.n_holes;

            if (fitter_state.count_missed() > 0u) {
                n_trk_missing_fw++;
            }
            if (fitter_state.n_holes > 0u) {
                //std::cout << "KF TRACK: " << i << std::endl;
                n_trk_holes_fw++;
            }
        }
        std::cout << "No. skipped states in fw KF: " << n_missed_fw
                  << std::endl;
        std::cout << "No. tracks with skipped states in fw KF: "
                  << n_trk_missing_fw << std::endl;
        std::cout << "No. holes found by fw KF: " << n_holes_fw << std::endl;
        std::cout << "No. tracks with holes in fw KF: " << n_trk_holes_fw
                  << std::endl;

        // Trigger failures
        if (n_miss_nav_fw.n_total() != n_missed_fw) {
            TRACCC_ERROR(
                "Forward filter number of missed states incorrect: was "
                << n_missed_fw << ", should be " << n_miss_nav_fw.n_total());
            test_successful = false;
        }
        if (n_trk_missing_fw != trk_stats_fw.n_tracks_w_holes) {
            TRACCC_ERROR(
                "Forward filter number of faulty tracks incorrect: was "
                << n_trk_missing_fw << ", should be "
                << trk_stats_fw.n_tracks_w_holes);
            test_successful = false;
        }
        if (n_miss_truth_fw.n_total() != n_holes_fw) {
            TRACCC_ERROR("Forward filter hole counting incorrect: was "
                         << n_holes_fw << ", should be "
                         << n_miss_truth_fw.n_total());
            test_successful = false;
        }
        if (n_trk_holes_fw < trk_stats_fw.n_tracks_w_extra) {
            TRACCC_ERROR(
                "Forward filter number of tracks with holes incorrect: was "
                << n_trk_holes_fw << ", should be "
                << trk_stats_fw.n_tracks_w_extra);
            test_successful = false;
        }

        std::cout << "-----------------------------------" << std::endl;
        std::cout << "BACKWARD - With KF" << std::endl
                  << "-----------------------------------\n";
        using fit_actor_bd =
            traccc::kalman_actor<algebra_t,
                                 kalman_actor_direction::BIDIRECTIONAL>;
        using actor_chain_bw_t =
            detray::actor_chain<transporter, fit_actor_bd, interactor, resetter,
                                perigee_stopper>;

        vecmem::vector<typename actor_chain_bw_t::state_tuple> state_tuple_bw{};
        vecmem::vector<typename actor_chain_bw_t::state_ref_tuple>
            state_ref_tuple_bw{};
        state_tuple_bw.reserve(tracks.size());
        state_ref_tuple_bw.reserve(tracks.size());

        // Prepare the fitter state for every track
        for (std::size_t i = 0u; i < tracks.size(); ++i) {
            auto& track_states = track_states_coll_bw[i];
            // Prepare actor states

            // Using a view here guarantees that the forward pass fits the
            // states and only the holes counter etc. gets reset when the
            // backward state is constructed
            auto trk_states_view = vecmem::get_data(track_states);
            fit_actor_bd::state fit_actor_state{
                vecmem::device_vector<track_state<algebra_t>>(trk_states_view)};
            fit_actor_state.do_precise_hole_count = false;

            state_tuple_bw.push_back(detray::make_tuple(
                fit_actor_state, interactor_state, resetter_state, stopper_state));
            state_ref_tuple_bw.push_back(
                setup_actor_states(state_tuple_bw.back()));
        }

        // Backward filter
        test_cfg.name(det.name(names) + "_GeV_bw_KF");
        test_cfg.navigation_direction(
            detray::navigation::direction::e_backward);
        const auto [trk_stats_bw, n_surfaces_bw, n_miss_nav_bw, n_miss_truth_bw,
                    step_traces_bw, mat_traces_bw, mat_records_bw] =
            detray::navigation_validator::compare_to_navigation<
                stepper_t, transporter, fit_actor_bd, interactor, resetter,
                perigee_stopper>(test_cfg, host_mr, det, names, ctx, field_view,
                                 prop_cfg, truth_traces_bw_KF, tracks,
                                 state_ref_tuple_bw, stddevs_per_track);

        // Check, how many tracks were smoothed correctly
        const auto n_tracks = static_cast<double>(trk_stats_bw.n_tracks);
        std::size_t n_missed_bw{0u};
        std::size_t n_holes_bw{0u};
        std::size_t n_trk_holes_bw{0u};
        std::size_t n_trk_missing_bw{0u};
        std::size_t n_not_smoothed_correctly{0u};
        for (std::size_t i = 0u; i < tracks.size(); ++i) {
            const auto& actor_states = state_tuple_bw[i];
            auto fitter_state = detray::get<fit_actor_fw::state>(actor_states);

            // What the actor counted
            n_missed_bw += fitter_state.count_missed();
            n_holes_bw += fitter_state.n_holes;

            if (fitter_state.count_missed() > 0u) {
                n_trk_missing_bw++;
            }
            if (fitter_state.n_holes > 0u) {
                n_trk_holes_bw++;
            }

            for (const auto& trk_state : fitter_state.m_track_states) {
                if (!trk_state.is_smoothed) {
                    n_not_smoothed_correctly++;
                    break;
                }
            }
        }
        std::cout << "INCLUDES MISSED STATES BY FW KF PASS:\n" << std::endl;
        std::cout << "No. skipped states in bw KF: " << n_missed_bw
                  << std::endl;
        std::cout << "No. tracks with skipped states in bw KF: "
                  << n_trk_missing_bw << std::endl;
        std::cout << "No. holes found by bw KF: " << n_holes_bw << std::endl;
        std::cout << "No. tracks with holes in bw KF: " << n_trk_holes_bw
                  << std::endl;
        std::cout << "No. tracks that were not smoothed correctly: "
                  << n_not_smoothed_correctly << " ("
                  << 100. * static_cast<double>(n_not_smoothed_correctly) /
                         n_tracks
                  << "%)" << std::endl;
        std::cout << "-----------------------------------" << std::endl;

        // Trigger failures
        if (n_missed_bw < n_miss_nav_bw.n_total()) {
            TRACCC_ERROR(
                "Backward filter number of missed states incorrect: was "
                << n_missed_bw << ", should be greater equal "
                << n_miss_nav_bw.n_total());
            test_successful = false;
        }
        if (n_trk_missing_bw < trk_stats_bw.n_tracks_w_holes) {
            TRACCC_ERROR(
                "Backward filter number of faulty tracks incorrect: was "
                << n_trk_missing_bw << ", should be greater equal "
                << trk_stats_bw.n_tracks_w_holes);
            test_successful = false;
        }
        if (n_miss_truth_bw.n_total() != n_holes_bw) {
            TRACCC_ERROR("Backward filter hole counting incorrect: was "
                         << n_holes_bw << ", should be "
                         << n_miss_truth_bw.n_total());
            test_successful = false;
        }
        if (n_trk_holes_bw != trk_stats_bw.n_tracks_w_extra) {
            TRACCC_ERROR(
                "Backward filter number of tracks with holes incorrect: was "
                << n_trk_holes_bw << ", should be "
                << trk_stats_bw.n_tracks_w_extra);
            test_successful = false;
        }
    }

    return test_successful;
}

}  // namespace traccc
