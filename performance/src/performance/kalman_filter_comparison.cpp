/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/bfield/construct_const_bfield.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/edm/track_state_collection.hpp"
#include "traccc/edm/track_state_helpers.hpp"
#include "traccc/fitting/kalman_filter/kalman_actor.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/host_detector.hpp"
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
    const traccc::host_detector* host_det,
    const traccc::default_detector::host::name_map& names,
    const traccc::magnetic_field& bfield,
    const detray::propagation::config& prop_cfg, const std::string& input_dir,
    const unsigned int n_events, std::unique_ptr<const traccc::Logger> ilogger,
    const bool do_multiple_scattering, const bool do_energy_loss,
    const bool use_acts_geoid,
    const traccc::pdg_particle<traccc::scalar> ptc_type,
    const traccc::seed_generator<traccc::default_detector::host>::config&
        smearing_cfg,
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

    // Retrieve detector
    const detector_t& det = host_det->template as<traccc::default_detector>();

    // Create B field
    b_field_t field = traccc::construct_const_bfield(B)
                          .as_field<traccc::const_bfield_backend_t<scalar_t>>();
    b_field_t::view_t field_view = field;

    // Seed smearing
    auto param_smearer =
        seed_generator{det, smearing_cfg, std::mt19937::default_seed, ctx};

    // Collect data for comparison

    // Initial track parameters from truth particle
    std::vector<traccc::free_track_parameters<algebra_t>> tracks{};
    // The traces of truth hits forward and in reverse order
    std::vector<vecmem::vector<sf_candidate_t>> truth_traces_fw{};
    std::vector<vecmem::vector<sf_candidate_t>> truth_traces_bw{};
    // Collection for bound track parameters and track state and measurement
    // links
    typename edm::track_fit_collection<algebra_t>::host track_param_coll{
        host_mr};
    // Measurments
    typename measurement_collection_types::host measurement_coll{};
    // Track states containing the result parameters for the KF
    typename edm::track_state_collection<algebra_t>::host track_state_coll{
        host_mr};
    typename edm::track_state_collection<algebra_t>::host track_states_coll_bw{
        host_mr};

    tracks.reserve(n_events * 1000);
    truth_traces_fw.reserve(tracks.capacity());
    truth_traces_bw.reserve(tracks.capacity());

    TRACCC_VERBOSE("Reconstructing " << n_events << " events");

    // Read the truth data in
    for (std::size_t i_event = 0u; i_event < n_events; ++i_event) {
        traccc::event_data evt_data(input_dir, i_event, host_mr, use_acts_geoid,
                                    host_det, data_format::csv);
        TRACCC_VERBOSE("Event " << i_event << ": Found "
                                << evt_data.m_particle_map.size()
                                << " initial particles");

        if (evt_data.m_particle_map.empty()) {
            TRACCC_ERROR("Removing event " << i_event
                                           << ": Found no particles");
            continue;
        }

        if (evt_data.m_ptc_to_meas_map.empty()) {
            TRACCC_ERROR(
                "Removing event "
                << i_event
                << ": Found no connections between particles and measurements");
            continue;
        }

        std::size_t n_tracks{tracks.size()};
        for (const auto& [ptc_id, ptc] : evt_data.m_particle_map) {
            if (!evt_data.m_ptc_to_meas_map.contains(ptc)) {
                continue;
            }
            // Minimum momentum
            const traccc::scalar pT{vector::perp(ptc.momentum)};
            if (pT <= min_pT) {
                TRACCC_WARNING("Event "
                               << i_event << ": Removing particle " << ptc_id
                               << " due to transv. momentum cut (pT was "
                               << pT / traccc::unit<traccc::scalar>::MeV
                               << " MeV)");
                continue;
            }

            // Make a trace of detray-understandable intersections
            auto truth_trace_fw =
                traccc::propagation_validator::transcribe_to_trace(
                    ctx, det, ptc, evt_data.m_ptc_to_meas_map);

            // No meansurements found
            if (truth_trace_fw.empty()) {
                TRACCC_WARNING("Event "
                               << i_event
                               << ": No measurements found for particle "
                               << ptc_id);
                continue;
            }

            // Minimum radius (remove secondaries)
            const traccc::scalar rad{vector::perp(ptc.vertex)};
            if (rad >= max_rad) {
                TRACCC_WARNING("Event "
                               << i_event << ": Removing particle " << ptc_id
                               << " due to radius cut (radius was "
                               << rad / traccc::unit<traccc::scalar>::mm
                               << " mm)");
                continue;
            }

            // Revert the forward trace for the backward propagation
            vecmem::vector<sf_candidate_t> truth_trace_bw(
                truth_trace_fw.size());
            std::ranges::reverse_copy(truth_trace_fw, truth_trace_bw.begin());

            assert(!truth_trace_bw.empty());

            truth_traces_fw.push_back(std::move(truth_trace_fw));
            truth_traces_bw.push_back(std::move(truth_trace_bw));

            TRACCC_DEBUG("Event " << i_event << ": Found "
                                  << truth_traces_fw.size()
                                  << " truth measurement(s) for track "
                                  << tracks.size());

            // Transcribe measurements to global collection
            const auto& measurements_per_ptc =
                evt_data.m_ptc_to_meas_map.at(ptc);
            assert(!measurements_per_ptc.empty());

            const auto meas_offset{
                static_cast<unsigned int>(measurement_coll.size())};
            measurement_coll.insert(std::end(measurement_coll),
                                    std::begin(measurements_per_ptc),
                                    std::end(measurements_per_ptc));

            measurement_collection_types::const_device measurement_view{
                vecmem::get_data(measurement_coll)};

            track_state_coll.reserve(track_state_coll.size() +
                                     measurements_per_ptc.size());

            // Construct initial track parameters for the propagation
            // @TODO: Need volume grid in case of large vertex smearing
            tracks.emplace_back(ptc.vertex, 0.f, ptc.momentum, ptc.charge);

            TRACCC_DEBUG("-> Truth track " << tracks.size() - 1u << ": "
                                           << tracks.back());

            // Connect the bound track parameters to the measurements and
            // states
            typename edm::track_fit_collection<algebra_t>::host::object_type
                track_object{};

            for (unsigned int i = 0u; i < measurements_per_ptc.size(); ++i) {
                unsigned int meas_idx{meas_offset + i};

                track_object.state_indices().push_back(meas_idx);

                track_state_coll.push_back(edm::make_track_state<algebra_t>(
                    measurement_view, meas_idx));
                track_states_coll_bw.push_back(edm::make_track_state<algebra_t>(
                    measurement_view, meas_idx));
            }

            track_param_coll.push_back(track_object);
        }

        if (track_param_coll.size() - n_tracks == 0u) {
            TRACCC_WARNING("Event " << i_event
                                    << ": No eligible tracks in event");
        } else {
            TRACCC_VERBOSE("Event "
                           << i_event << ": Found "
                           << track_param_coll.size() - n_tracks
                           << " reconstructible truth track(s) in event");
        }
    }

    // Check truth data
    if (truth_traces_fw.empty() || truth_traces_bw.empty()) {
        TRACCC_ERROR("Propagation truth data empty");
        return false;
    }
    if (track_state_coll.size() == 0u) {
        TRACCC_ERROR("Kalman Filter truth data empty");
        return false;
    }

    using transporter = detray::parameter_transporter<algebra_t>;
    using interactor = detray::pointwise_material_interactor<algebra_t>;
    using resetter = detray::parameter_resetter<algebra_t>;

    // Run the navigation and compare
    detray::test::navigation_validation_config<algebra_t> test_cfg{};
    test_cfg.n_tracks(tracks.size()).ptc_hypothesis(ptc_type);
    test_cfg.collect_sensitives_only(true).fail_on_diff(false);
    test_cfg.display_only_missed(true).verbose(false);

    // Make a tuple of references from a tuple
    auto setup_actor_states = []<typename... T>(detray::dtuple<T...>& t) {
        return detray::tie(detray::detail::get<T>(t)...);
    };

    // Safe the original truth traces before dummy records are inserted for
    // missing intersections
    auto truth_traces_fw_KF = truth_traces_fw;
    auto truth_traces_bw_KF = truth_traces_bw;

    // Reusable actor states
    resetter::state resetter_state{};
    resetter_state.n_stddev = prop_cfg.navigation.n_scattering_stddev;
    resetter_state.accumulated_error = prop_cfg.navigation.accumulated_error;
    resetter_state.estimate_scattering_noise =
        prop_cfg.navigation.estimate_scattering_noise;
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
                truth_traces_fw, tracks, state_ref_tuples, param_smearer);

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
                param_smearer);

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

        auto track_param_view{vecmem::get_data(track_param_coll)};
        typename edm::track_fit_collection<algebra_t>::device
            track_param_device_container{track_param_view};
        auto track_state_view{vecmem::get_data(track_state_coll)};
        typename edm::track_state_collection<algebra_t>::device
            track_state_device_container{track_state_view};
        typename measurement_collection_types::const_device
            measuremen_device_container{vecmem::get_data(measurement_coll)};

        vecmem::vector<typename actor_chain_t::state_tuple> state_tuple{};
        vecmem::vector<typename actor_chain_t::state_ref_tuple>
            state_ref_tuple{};
        state_tuple.reserve(tracks.size());
        state_ref_tuple.reserve(tracks.size());

        // Prepare the fitter state for every track
        for (std::size_t i = 0u; i < tracks.size(); ++i) {

            // Prepare actor states
            fit_actor_fw::state fit_actor_state(
                track_param_device_container.at(static_cast<unsigned int>(i)),
                track_state_device_container, measuremen_device_container);
            fit_actor_state.do_precise_hole_count = true;
            fit_actor_state.reset();

            state_tuple.push_back(detray::make_tuple(
                interactor_state, fit_actor_state, resetter_state));
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
                truth_traces_fw_KF, tracks, state_ref_tuple, param_smearer);

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
                // std::cout << "KF TRACK: " << i << std::endl;
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

        /*std::cout << "-----------------------------------" << std::endl;
        std::cout << "BACKWARD - With KF" << std::endl
                  << "-----------------------------------\n";
        using fit_actor_bd =
            traccc::kalman_actor<algebra_t,
                                 kalman_actor_direction::BIDIRECTIONAL>;
        using perigee_stopper = detray::perigee_stopper<algebra_t>;
        using actor_chain_bw_t =
            detray::actor_chain<transporter, fit_actor_bd, interactor, resetter,
                                perigee_stopper>;

        auto track_state_bw_view{vecmem::get_data(track_states_coll_bw)};
        auto track_state_device_container_bw =
            typename
        edm::track_state_collection<algebra_t>::device{track_state_bw_view};

        vecmem::vector<typename actor_chain_bw_t::state_tuple> state_tuple_bw{};
        vecmem::vector<typename actor_chain_bw_t::state_ref_tuple>
            state_ref_tuple_bw{};
        state_tuple_bw.reserve(tracks.size());
        state_ref_tuple_bw.reserve(tracks.size());

        perigee_stopper::state stopper_state{};

        // Prepare the fitter state for every track
        for (std::size_t i = 0u; i < tracks.size(); ++i) {
            // Prepare actor states

            // Using a view here guarantees that the forward pass fits the
            // states and only the holes counter etc. gets reset when the
            // backward state is constructed
            fit_actor_bd::state fit_actor_state(
                track_param_device_container.at(static_cast<unsigned int>(i)),
                track_state_device_container_bw, measuremen_device_container);
            fit_actor_state.do_precise_hole_count = true;
            fit_actor_state.reset();

            state_tuple_bw.push_back(
                detray::make_tuple(fit_actor_state, interactor_state,
                                   resetter_state, stopper_state));
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
                                 state_ref_tuple_bw, param_smearer);

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

            for (unsigned int istate = 0u; istate < fitter_state.size();
                 ++istate) {
                if (!fitter_state.at(istate).is_smoothed()) {
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
        }*/
    }

    return test_successful;
}

}  // namespace traccc
