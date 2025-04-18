/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/edm/track_parameters.hpp"
#include "traccc/fitting/kalman_filter/kalman_actor.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/data_format.hpp"
#include "traccc/simulation/simulator.hpp"
#include "traccc/utils/event_data.hpp"
#include "traccc/utils/navigation_validation.hpp"

// Test include(s).
#include "tests/toy_detector_fixture.hpp"

// Detray include(s).
#include <detray/detectors/bfield.hpp>
#include <detray/io/frontend/detector_reader.hpp>
#include <detray/propagator/actors.hpp>
#include <detray/propagator/rk_stepper.hpp>

// Detray test include(s)
#include <detray/test/utils/simulation/event_generator/track_generators.hpp>
#include <detray/test/validation/navigation_validation_utils.hpp>

// GTest include(s).
#include <gtest/gtest.h>

using namespace traccc;

constexpr double rel_mat_error{0.01};
constexpr std::size_t n_events{5u};
constexpr detray::pdg_particle ptc_type{detray::muon<scalar>()};

/// Test suite for navigation tests for the CKF
class CKF_navigation_test
    : public ToyDetectorFixture,
      public ::testing::WithParamInterface<std::tuple<
          float, float, float, float, std::size_t, bool, bool, bool>> {};

/// Test the detray navigation on simulated tracks
TEST_P(CKF_navigation_test, toy_detector) {

    using algebra_t = traccc::default_algebra;
    using detector_t = traccc::default_detector::host;
    using b_field_t = covfie::field<detray::bfield::const_bknd_t<scalar>>;
    using track_t = traccc::free_track_parameters<algebra_t>;
    using sf_candidate_t =
        traccc::navigation_validator::candidate_type<detector_t>;

    using generator_t = detray::random_track_generator<track_t>;
    using writer_t = smearing_writer<measurement_smearer<algebra_t>>;
    using simulator_t = simulator<detector_t, b_field_t, generator_t, writer_t>;

    using stepper_t = detray::rk_stepper<
        b_field_t::view_t, algebra_t, detray::constrained_step<scalar>,
        detray::stepper_rk_policy<scalar>, detray::stepping::print_inspector>;

    vecmem::host_memory_resource host_mr;

    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file("toy_detector_geometry.json")
        .add_file("toy_detector_surface_grids.json")
        .do_check(true);
    if (std::get<5>(GetParam())) {
        reader_cfg.add_file("toy_detector_homogeneous_material.json");
    }

    const auto [det, names] =
        detray::io::read_detector<traccc::default_detector::host>(host_mr,
                                                                  reader_cfg);

    // Create toy detector
    const detector_t::geometry_context ctx{};

    // Create B field
    auto field = detray::bfield::create_const_field<scalar>(B);
    b_field_t::view_t field_view = field;

    // Create track generator
    const scalar pT{std::get<0>(GetParam())};
    generator_t::configuration gen_cfg{};
    gen_cfg.n_tracks(2500u).eta_range(-3, 3).p_T(pT).randomize_charge(true);
    // Choose different random seed than detray for more test coverage
    gen_cfg.seed(135346);

    // Create measurement smearer
    measurement_smearer<algebra_t> smearer(smearing[0], smearing[1]);
    auto sim = simulator_t(ptc_type, n_events, det, field, generator_t{gen_cfg},
                           writer_t::config{smearer},
                           "../data/detray_simulation/toy_detector");

    // Propagation config for the simulation
    detray::propagation::config prop_cfg{};
    prop_cfg.navigation.search_window = search_window;  //< toy detector grids

    sim.get_config().propagation = prop_cfg;
    sim.get_config().do_multiple_scattering = std::get<6>(GetParam());
    sim.get_config().do_energy_loss = std::get<7>(GetParam());
    sim.get_config().min_pT(10.f * traccc::unit<scalar>::MeV);

    // Do the simulation: Produces data files
    sim.run();

    // Specific config for the navigation test
    prop_cfg.navigation.min_mask_tolerance = std::get<1>(GetParam());
    prop_cfg.navigation.mask_tolerance_scalor = 1.f;
    prop_cfg.navigation.overstep_tolerance = -1000.f * traccc::unit<float>::um;
    prop_cfg.navigation.max_mask_tolerance =
        std::get<1>(GetParam()) + 3.f * traccc::unit<float>::mm;

    // Collect data for comparison

    // Initial track parameters
    std::vector<traccc::free_track_parameters<algebra_t>> tracks{};
    // The traces of truth hits forward and in reverse order
    std::vector<vecmem::vector<sf_candidate_t>> truth_traces_fw{};
    std::vector<vecmem::vector<sf_candidate_t>> truth_traces_bw{};
    std::vector<vecmem::vector<track_state<algebra_t>>> track_state_coll{};

    tracks.reserve(n_events * gen_cfg.n_tracks());
    truth_traces_fw.reserve(tracks.capacity());
    truth_traces_bw.reserve(tracks.capacity());

    // Read the simulation data back in
    for (std::size_t i_event = 0u; i_event < n_events; i_event++) {
        traccc::event_data evt_data("detray_simulation/toy_detector", i_event,
                                    host_mr, false, &det, data_format::csv);

        ASSERT_FALSE(evt_data.m_particle_map.empty());
        ASSERT_FALSE(evt_data.m_ptc_to_meas_map.empty());
        ASSERT_EQ(evt_data.m_particle_map.size(), gen_cfg.n_tracks());

        for (const auto& [ptc_id, ptc] : evt_data.m_particle_map) {
            // Make a trace of detray-understandable intersections
            auto truth_trace_fw =
                traccc::navigation_validator::transcribe_to_trace(
                    ctx, det, ptc, evt_data.m_ptc_to_meas_map);

            // Revert the forward trace fro the backward propagation
            vecmem::vector<sf_candidate_t> truth_trace_bw(
                truth_trace_fw.size());
            std::ranges::reverse_copy(truth_trace_fw, truth_trace_bw.begin());

            if (!truth_trace_fw.empty()) {
                // Construct initial track parameters
                // @TODO: Need volume grid in case of large vertex smearing
                tracks.emplace_back(ptc.vertex, 0.f, ptc.momentum, ptc.charge);

                truth_traces_fw.push_back(std::move(truth_trace_fw));
                truth_traces_bw.push_back(std::move(truth_trace_bw));
            }

            // Transcribe measurements to track states for the KF
            if (!evt_data.m_ptc_to_meas_map.contains(ptc) ||
                vector::norm(ptc.momentum) < 50.f * traccc::unit<scalar>::MeV) {
                continue;
            }
            const auto& measurements = evt_data.m_ptc_to_meas_map.at(ptc);
            vecmem::vector<track_state<algebra_t>> track_states{};
            track_states.reserve(measurements.size());

            for (const auto& meas : measurements) {
                track_states.emplace_back(meas);
            }

            track_state_coll.push_back(std::move(track_states));
        }
    }

    ASSERT_EQ(n_events * gen_cfg.n_tracks(), tracks.size());
    ASSERT_EQ(truth_traces_fw.size(), tracks.size());
    ASSERT_EQ(truth_traces_bw.size(), tracks.size());
    ASSERT_EQ(track_state_coll.size(), tracks.size());

    using perigee_stopper = detray::perigee_stopper<algebra_t>;
    using transporter = detray::parameter_transporter<algebra_t>;
    using interactor = detray::pointwise_material_interactor<algebra_t>;
    using resetter = detray::parameter_resetter<algebra_t>;

    // Run the navigation and compare
    detray::test::navigation_validation_config test_cfg{};
    test_cfg.n_tracks(tracks.size()).ptc_hypothesis(ptc_type);
    test_cfg.collect_sensitives_only(true).fail_on_diff(false);
    test_cfg.display_only_missed(true).verbose(false);

    // Make a tuple of references from a tuple
    auto setup_actor_states = []<typename... T>(detray::dtuple<T...> & t) {
        return detray::tie(detray::detail::get<T>(t)...);
    };

    auto truth_traces_fw_KF = truth_traces_fw;
    auto truth_traces_bw_KF = truth_traces_bw;

    // Initial state smearing
    vecmem::vector<std::array<scalar, e_bound_size>> stddevs_per_track{};

    perigee_stopper::state stopper_state{};
    interactor::state interactor_state{};
    interactor_state.do_multiple_scattering = std::get<6>(GetParam());
    interactor_state.do_energy_loss = std::get<7>(GetParam());

    {
        std::cout << "-----------------------------------"
                  << "\nFORWARD - No KF" << std::endl
                  << "-----------------------------------\n";

        // Prepare actor states
        auto state_tuple = detray::make_tuple(interactor_state);
        auto state_ref_tuple = vecmem::vector{setup_actor_states(state_tuple)};

        // Forward navigation
        test_cfg.name(std::to_string(pT) + "_GeV_fw");
        test_cfg.navigation_direction(detray::navigation::direction::e_forward);
        const auto [trk_stats_fw, n_surfaces_fw, n_miss_nav_fw, n_miss_truth_fw,
                    step_traces_fw, mat_traces_fw, mat_records_fw] =
            detray::navigation_validator::compare_to_navigation<
                stepper_t, transporter, interactor, resetter>(
                test_cfg, host_mr, det, names, ctx, field_view, prop_cfg,
                truth_traces_fw, tracks, state_ref_tuple);

        std::cout << "BACKWARD - No KF" << std::endl
                  << "-----------------------------------\n";
        auto bw_state_tuple =
            detray::make_tuple(interactor_state, stopper_state);
        auto bw_state_ref_tuple =
            vecmem::vector{setup_actor_states(bw_state_tuple)};

        // Backward navigation
        test_cfg.name(std::to_string(pT) + "_GeV_bw");
        test_cfg.navigation_direction(
            detray::navigation::direction::e_backward);
        const auto [trk_stats_bw, n_surfaces_bw, n_miss_nav_bw, n_miss_truth_bw,
                    step_traces_bw, mat_traces_bw, mat_records_bw] =
            detray::navigation_validator::compare_to_navigation<
                stepper_t, transporter, interactor, resetter, perigee_stopper>(
                test_cfg, host_mr, det, names, ctx, field_view, prop_cfg,
                truth_traces_bw, tracks, bw_state_ref_tuple);

        // Make sure some data was collected
        ASSERT_TRUE(trk_stats_fw.n_tracks > 0u);
        ASSERT_TRUE(n_surfaces_fw.n_total() > 0u);
        ASSERT_TRUE(trk_stats_bw.n_tracks > 0u);
        ASSERT_TRUE(n_surfaces_bw.n_total() > 0u);

        ASSERT_EQ(trk_stats_fw.n_tracks, trk_stats_bw.n_tracks);

        // Check, the amount of collected material between forward and backward
        ASSERT_EQ(step_traces_fw.size(), trk_stats_fw.n_tracks);
        ASSERT_EQ(mat_traces_fw.size(), trk_stats_fw.n_tracks);
        ASSERT_EQ(mat_records_fw.size(), trk_stats_fw.n_tracks);
        ASSERT_EQ(mat_records_fw.size(), mat_records_bw.size());
        ASSERT_EQ(step_traces_fw.size(), step_traces_bw.size());
        ASSERT_EQ(mat_traces_fw.size(), mat_traces_bw.size());

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

            // Revert the backward trace to comapre to the forward trace
            std::remove_cvref_t<decltype(mat_traces_bw[i])> inv_mat_trace_bw(
                mat_traces_bw[i].size());
            std::ranges::reverse_copy(mat_traces_bw[i],
                                      inv_mat_trace_bw.begin());

            ASSERT_EQ(mat_traces_bw[i].size(), inv_mat_trace_bw.size());
            ASSERT_EQ(mat_traces_bw[i].front().bcd,
                      inv_mat_trace_bw.back().bcd);

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

        EXPECT_EQ(n_diff_mat, 0u);

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
                    std::get<4>(GetParam()));
    }

    {
        std::cout << "-----------------------------------"
                  << "\nFORWARD - With KF" << std::endl
                  << "-----------------------------------\n";

        using fit_actor = traccc::kalman_actor<algebra_t>;
        using actor_chain_t =
            detray::actor_chain<transporter, interactor, fit_actor, resetter>;

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
            fit_actor::state fit_actor_state{
                vecmem::device_vector<track_state<algebra_t>>(trk_states_view)};

            state_tuple.push_back(
                detray::make_tuple(interactor_state, fit_actor_state));
            state_ref_tuple.push_back(setup_actor_states(state_tuple.back()));
            stddevs_per_track.push_back(stddevs);
        }

        // Forward filter
        test_cfg.name(std::to_string(pT) + "_GeV_fw_KF");
        test_cfg.navigation_direction(detray::navigation::direction::e_forward);
        const auto [trk_stats_fw, n_surfaces_fw, n_miss_nav_fw, n_miss_truth_fw,
                    step_traces_fw, mat_traces_fw, mat_records_fw] =
            detray::navigation_validator::compare_to_navigation<
                stepper_t, transporter, interactor, fit_actor, resetter>(
                test_cfg, host_mr, det, names, ctx, field_view, prop_cfg,
                truth_traces_fw_KF, tracks, state_ref_tuple, stddevs_per_track);

        // Check, how many tracks were smoothed correctly
        auto n_tracks{static_cast<double>(trk_stats_fw.n_tracks)};
        std::size_t n_holes_fw{0u};
        for (std::size_t i = 0u; i < tracks.size(); ++i) {
            const auto& actor_states = state_tuple[i];
            auto fitter_state = detray::get<fit_actor::state>(actor_states);

            for (const auto& trk_state : fitter_state.m_track_states) {
                if (trk_state.is_hole) {
                    n_holes_fw++;
                }
            }
        }
        std::cout << "No. holes found by fw KF: " << n_holes_fw << std::endl;
        std::cout << "-----------------------------------" << std::endl;

        std::cout << "BACKWARD - With KF" << std::endl
                  << "-----------------------------------\n";
        using actor_chain_bw_t =
            detray::actor_chain<transporter, fit_actor, interactor, resetter,
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
            auto trk_states_view = vecmem::get_data(track_states);
            fit_actor::state fit_actor_state{
                vecmem::device_vector<track_state<algebra_t>>(trk_states_view)};

            state_tuple_bw.push_back(detray::make_tuple(
                fit_actor_state, interactor_state, stopper_state));
            state_ref_tuple_bw.push_back(
                setup_actor_states(state_tuple_bw.back()));
        }

        // Backward filter
        test_cfg.name(std::to_string(pT) + "_GeV_bw_KF");
        test_cfg.navigation_direction(
            detray::navigation::direction::e_backward);
        const auto [trk_stats_bw, n_surfaces_bw, n_miss_nav_bw, n_miss_truth_bw,
                    step_traces_bw, mat_traces_bw, mat_records_bw] =
            detray::navigation_validator::compare_to_navigation<
                stepper_t, transporter, fit_actor, interactor, resetter,
                perigee_stopper>(test_cfg, host_mr, det, names, ctx, field_view,
                                 prop_cfg, truth_traces_bw_KF, tracks,
                                 state_ref_tuple_bw, stddevs_per_track);

        // Check, how many tracks were smoothed correctly
        n_tracks = static_cast<double>(trk_stats_bw.n_tracks);
        std::size_t n_holes_bw{0u};
        std::size_t n_not_smoothed_correctly{0u};
        for (std::size_t i = 0u; i < tracks.size(); ++i) {
            const auto& actor_states = state_tuple_bw[i];
            auto fitter_state = detray::get<fit_actor::state>(actor_states);

            // n_holes_bw += fitter_state.n_holes;
            for (const auto& trk_state : fitter_state.m_track_states) {
                if (trk_state.is_hole) {
                    n_holes_bw++;
                }
            }

            for (const auto& trk_state : fitter_state.m_track_states) {
                if (!trk_state.is_smoothed) {
                    n_not_smoothed_correctly++;
                    break;
                }
            }
        }
        std::cout << "No. holes found by bw KF: " << n_holes_bw << std::endl;
        std::cout << "No. tracks that were not smoothed correctly: "
                  << n_not_smoothed_correctly << " ("
                  << 100. * static_cast<double>(n_not_smoothed_correctly) /
                         n_tracks
                  << "%)" << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }
}

// Parameters:
// 1: p_T
// 2: min mask tolerance
// 3: max allowed % of track with holes
// 4: max allowed % of track with extra surfaces
// 5: max allowed number of holes per track
// 6: Build detector with material
// 7: Do multiple scattering
// 8: Do energy loss

// No material - navigation should work
INSTANTIATE_TEST_SUITE_P(
    pT_100GeV_no_mat, CKF_navigation_test,
    ::testing::Values(std::make_tuple(100.f * traccc::unit<scalar>::GeV,
                                      1e-5f * traccc::unit<float>::mm, 0.001f,
                                      0.001f, 1u, false, false, false)));

INSTANTIATE_TEST_SUITE_P(
    pT_50GeV_no_mat, CKF_navigation_test,
    ::testing::Values(std::make_tuple(50.f * traccc::unit<scalar>::GeV,
                                      1e-5f * traccc::unit<float>::mm, 0.001f,
                                      0.001f, 1u, false, false, false)));

INSTANTIATE_TEST_SUITE_P(
    pT_10GeV_no_mat, CKF_navigation_test,
    ::testing::Values(std::make_tuple(10.f * traccc::unit<scalar>::GeV,
                                      1e-5f * traccc::unit<float>::mm, 0.001f,
                                      0.001f, 1u, false, false, false)));
INSTANTIATE_TEST_SUITE_P(
    pT_5GeV_no_mat, CKF_navigation_test,
    ::testing::Values(std::make_tuple(5.f * traccc::unit<scalar>::GeV,
                                      1e-5f * traccc::unit<float>::mm, 0.001f,
                                      0.001f, 1u, false, false, false)));

INSTANTIATE_TEST_SUITE_P(
    pT_1GeV_no_mat, CKF_navigation_test,
    ::testing::Values(std::make_tuple(1.f * traccc::unit<scalar>::GeV,
                                      1e-5f * traccc::unit<float>::mm, 0.001f,
                                      0.001f, 1u, false, false, false)));

INSTANTIATE_TEST_SUITE_P(
    pT_05GeV_no_mat, CKF_navigation_test,
    ::testing::Values(std::make_tuple(0.5f * traccc::unit<scalar>::GeV,
                                      1e-5f * traccc::unit<float>::mm, 0.001f,
                                      0.001f, 1u, false, false, false)));

INSTANTIATE_TEST_SUITE_P(
    pT_01GeV_no_mat, CKF_navigation_test,
    ::testing::Values(std::make_tuple(0.1 * traccc::unit<scalar>::GeV,
                                      1e-5f * traccc::unit<float>::mm, 0.001f,
                                      0.001f, 1u, false, false, false)));

// No scattering - navigation should work (material interactor models e-loss)
INSTANTIATE_TEST_SUITE_P(
    pT_100GeV_only_eloss, CKF_navigation_test,
    ::testing::Values(std::make_tuple(100.f * traccc::unit<scalar>::GeV,
                                      1e-3f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, false, true)));
INSTANTIATE_TEST_SUITE_P(
    pT_50GeV_only_eloss, CKF_navigation_test,
    ::testing::Values(std::make_tuple(50.f * traccc::unit<scalar>::GeV,
                                      1e-3f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, false, true)));
INSTANTIATE_TEST_SUITE_P(
    pT_10GeV_only_eloss, CKF_navigation_test,
    ::testing::Values(std::make_tuple(10.f * traccc::unit<scalar>::GeV,
                                      1e-3f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, false, true)));
INSTANTIATE_TEST_SUITE_P(
    pT_5GeV_only_eloss, CKF_navigation_test,
    ::testing::Values(std::make_tuple(5.f * traccc::unit<scalar>::GeV,
                                      1e-3f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, false, true)));

INSTANTIATE_TEST_SUITE_P(
    pT_1GeV_only_eloss, CKF_navigation_test,
    ::testing::Values(std::make_tuple(1.f * traccc::unit<scalar>::GeV,
                                      1e-3f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, false, true)));

INSTANTIATE_TEST_SUITE_P(
    pT_05GeV_only_eloss, CKF_navigation_test,
    ::testing::Values(std::make_tuple(0.5f * traccc::unit<scalar>::GeV,
                                      1e-3f * traccc::unit<float>::mm, 0.005f,
                                      0.005f, 4u, true, false, true)));

INSTANTIATE_TEST_SUITE_P(
    pT_01GeV_only_eloss, CKF_navigation_test,
    ::testing::Values(std::make_tuple(0.1f * traccc::unit<scalar>::GeV,
                                      1e-3f * traccc::unit<float>::mm, 0.005f,
                                      0.005f, 4u, true, false, true)));

// No energy loss - navigation has to compensate the scattering angle
// (turn off the material in the detector to prevent bethe-bloch corrections)
INSTANTIATE_TEST_SUITE_P(
    pT_100GeV_only_scatt, CKF_navigation_test,
    ::testing::Values(std::make_tuple(100.f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, true, false)));

INSTANTIATE_TEST_SUITE_P(
    pT_50GeV_only_scatt, CKF_navigation_test,
    ::testing::Values(std::make_tuple(50.f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, true, false)));

INSTANTIATE_TEST_SUITE_P(
    pT_10GeV_only_scatt, CKF_navigation_test,
    ::testing::Values(std::make_tuple(10.f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, true, false)));

INSTANTIATE_TEST_SUITE_P(
    pT_5GeV_only_scatt, CKF_navigation_test,
    ::testing::Values(std::make_tuple(5.f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, true, false)));

INSTANTIATE_TEST_SUITE_P(
    pT_1GeV_only_scatt, CKF_navigation_test,
    ::testing::Values(std::make_tuple(1.f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, true, false)));

INSTANTIATE_TEST_SUITE_P(
    pT_05GeV_only_scatt, CKF_navigation_test,
    ::testing::Values(std::make_tuple(0.5f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.005f,
                                      0.005f, 4u, true, true, false)));

INSTANTIATE_TEST_SUITE_P(
    pT_01GeV_only_scatt, CKF_navigation_test,
    ::testing::Values(std::make_tuple(0.1f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.005f,
                                      0.005f, 4u, true, true, false)));

// Nominal (e-loss + scattering)
INSTANTIATE_TEST_SUITE_P(
    pT_100GeV_nominal, CKF_navigation_test,
    ::testing::Values(std::make_tuple(100.f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.01f,
                                      0.15f, 1u, true, true, true)));
INSTANTIATE_TEST_SUITE_P(
    pT_50GeV_nominal, CKF_navigation_test,
    ::testing::Values(std::make_tuple(50.f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.01f,
                                      0.15f, 1u, true, true, true)));
INSTANTIATE_TEST_SUITE_P(
    pT_10GeV_nominal, CKF_navigation_test,
    ::testing::Values(std::make_tuple(10.f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.01f,
                                      0.15f, 1u, true, true, true)));
INSTANTIATE_TEST_SUITE_P(
    pT_5GeV_nominal, CKF_navigation_test,
    ::testing::Values(std::make_tuple(5.f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.01f,
                                      0.15f, 1u, true, true, true)));

INSTANTIATE_TEST_SUITE_P(
    pT_1GeV_nominal, CKF_navigation_test,
    ::testing::Values(std::make_tuple(1.f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.01f,
                                      0.5f, 3u, true, true, true)));

INSTANTIATE_TEST_SUITE_P(
    pT_05GeV_nominal, CKF_navigation_test,
    ::testing::Values(std::make_tuple(0.5f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.01f,
                                      0.5f, 3u, true, true, true)));

INSTANTIATE_TEST_SUITE_P(
    pT_01GeV_nominal, CKF_navigation_test,
    ::testing::Values(std::make_tuple(0.1f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.01f,
                                      0.5f, 3u, true, true, true)));
