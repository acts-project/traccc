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
#include "traccc/io/csv/make_hit_reader.hpp"
#include "traccc/io/csv/make_particle_reader.hpp"
#include "traccc/simulation/simulator.hpp"

// Test include(s).
#include "tests/navigation_validation.hpp"

// Detray include(s).
#include <detray/detectors/bfield.hpp>
#include <detray/propagator/actors.hpp>
#include <detray/propagator/rk_stepper.hpp>

// Detray test include(s)
#include <detray/test/utils/detectors/build_toy_detector.hpp>
#include <detray/test/utils/simulation/event_generator/track_generators.hpp>
#include <detray/test/validation/navigation_validation_utils.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <filesystem>
#include <stdexcept>

using namespace traccc;

constexpr scalar tol{1e-7f};

/// Test the detray navigation on simulated tracks
GTEST_TEST(traccc_ckf, toy_detector_navigation) {

    using algebra_t = traccc::default_algebra;
    using detector_t = traccc::toy_detector::host;
    using b_field_t = covfie::field<detray::bfield::const_bknd_t<scalar>>;
    using track_t = traccc::free_track_parameters<algebra_t>;

    using generator_t = detray::random_track_generator<track_t>;
    using writer_t = smearing_writer<measurement_smearer<algebra_t>>;
    using simulator_t = simulator<detector_t, b_field_t, generator_t, writer_t>;

    using stepper_t = detray::rk_stepper<
        b_field_t::view_t, algebra_t, detray::unconstrained_step<scalar>,
        detray::stepper_rk_policy<scalar>, detray::stepping::print_inspector>;

    vecmem::host_memory_resource host_mr;

    // Configuration
    constexpr std::size_t n_events{10u};
    constexpr std::size_t n_tracks_per_evt{2500u};
    constexpr scalar pT{5.f * traccc::unit<scalar>::GeV};
    constexpr scalar hit_var_x{50.f * traccc::unit<scalar>::um};
    constexpr scalar hit_var_y{50.f * traccc::unit<scalar>::um};
    detray::propagation::config prop_cfg{};
    prop_cfg.navigation.search_window = {3u, 3u};  //< toy detector grid search

    // Create toy detector
    const detector_t::geometry_context ctx{};

    detray::toy_det_config<scalar> toy_cfg{};
    toy_cfg.n_edc_layers(7u).use_material_maps(false).envelope(
        2.f * traccc::unit<scalar>::mm);

    auto [det, names] = detray::build_toy_detector<algebra_t>(host_mr, toy_cfg);

    // Create B field
    const vector3 B{0.f, 0.f, 2.f * traccc::unit<scalar>::T};
    auto field = detray::bfield::create_const_field<scalar>(B);

    // Create track generator
    generator_t::configuration gen_cfg{};
    gen_cfg.n_tracks(n_tracks_per_evt).p_T(pT);

    generator_t generator(gen_cfg);

    // Create measurement smearer
    measurement_smearer<algebra_t> smearer(hit_var_x, hit_var_y);

    writer_t::config writer_cfg{smearer};

    auto sim = simulator_t(detray::muon<scalar>(), n_events, det, field,
                           std::move(generator), std::move(writer_cfg));

    sim.get_config().propagation = prop_cfg;
    // sim.get_config().propagation.stepping.step_constraint =
    //    1.f * traccc::unit<scalar>::mm;

    // Do the simulation
    sim.run();

    // Specific config for the navigation test
    prop_cfg.navigation.min_mask_tolerance = 0.5f * traccc::unit<float>::mm;
    prop_cfg.navigation.mask_tolerance_scalor = 1.f;
    prop_cfg.navigation.max_mask_tolerance = 7.f * traccc::unit<float>::mm;
    prop_cfg.navigation.overstep_tolerance = -300.f * traccc::unit<float>::um;
    // prop_cfg.navigation.scale_tolerance = true;

    // Initial track parameters
    std::vector<traccc::free_track_parameters<algebra_t>> tracks{};
    // The traces of truth hits forward and in reverse order
    std::vector<
        dvector<traccc::navigation_validator::candidate_type<detector_t>>>
        truth_traces_fw{};
    std::vector<
        dvector<traccc::navigation_validator::candidate_type<detector_t>>>
        truth_traces_bw{};

    tracks.reserve(n_events * n_tracks_per_evt);
    truth_traces_fw.reserve(n_events * n_tracks_per_evt);
    truth_traces_bw.reserve(n_events * n_tracks_per_evt);

    for (std::size_t i_event = 0u; i_event < n_events; i_event++) {

        std::vector<traccc::io::csv::particle> particles;
        auto particle_reader = traccc::io::csv::make_particle_reader(
            traccc::io::get_event_filename(i_event, "-particles_initial.csv"));
        traccc::io::csv::particle io_particle;
        while (particle_reader.read(io_particle)) {
            particles.push_back(io_particle);
        }

        std::vector<traccc::io::csv::hit> hits;
        auto hit_reader = traccc::io::csv::make_hit_reader(
            traccc::io::get_event_filename(i_event, "-hits.csv"));
        traccc::io::csv::hit io_hit;
        while (hit_reader.read(io_hit)) {
            hits.push_back(io_hit);
        }

        ASSERT_EQ(particles.size(), gen_cfg.n_tracks());
        ASSERT_FALSE(hits.empty());

        // Run the navigation on every truth particle
        for (const auto& ptcl : particles) {

            // @TODO: Need volume grid in case of large vertex smearing
            const point3 vertex{ptcl.vx, ptcl.vy, ptcl.vz};
            const vector3 p{ptcl.px, ptcl.py, ptcl.pz};
            tracks.emplace_back(vertex, 0.f, p, ptcl.q);

            // Make a trace of detray-understandable hits
            auto truth_trace_fw =
                traccc::navigation_validator::transcribe_to_trace(
                    ctx, det, ptcl, hits,
                    detray::navigation::direction::e_forward);
            truth_traces_fw.push_back(std::move(truth_trace_fw));

            auto truth_trace_bw =
                traccc::navigation_validator::transcribe_to_trace(
                    ctx, det, ptcl, hits,
                    detray::navigation::direction::e_backward);
            truth_traces_bw.push_back(std::move(truth_trace_bw));
        }
    }

    using transporter = detray::parameter_transporter<algebra_t>;
    using interactor = detray::pointwise_material_interactor<algebra_t>;
    // using fit_actor = traccc::kalman_actor<algebra_t>;
    using resetter = detray::parameter_resetter<algebra_t>;

    // Run the navigation and compare
    constexpr bool collect_only_sensitives{true};
    constexpr bool fail_on_diff{false};
    constexpr bool verbose{false};

    b_field_t::view_t field_view = field;

    std::cout << "\n=========================" << std::endl
              << "|| FORWARD NAVIGATION  ||" << std::endl
              << "=========================\n";

    // Forward navigation
    auto nav_dir{detray::navigation::direction::e_forward};
    const auto [trk_stats_fw, n_surfaces_fw, n_miss_nav_fw, n_miss_truth_fw] =
        detray::navigation_validator::compare_to_navigation<
            stepper_t, transporter, interactor, resetter>(
            host_mr, det, names, ctx, field_view, prop_cfg, truth_traces_fw,
            tracks, nav_dir, collect_only_sensitives, fail_on_diff, verbose);

    std::cout << "\n=========================" << std::endl
              << "|| BACKWARD NAVIGATION ||" << std::endl
              << "=========================\n";

    // Backward navigation
    nav_dir = detray::navigation::direction::e_backward;
    const auto [trk_stats_bw, n_surfaces_bw, n_miss_nav_bw, n_miss_truth_bw] =
        detray::navigation_validator::compare_to_navigation<
            stepper_t, transporter, interactor, resetter>(
            host_mr, det, names, ctx, field_view, prop_cfg, truth_traces_bw,
            tracks, nav_dir, collect_only_sensitives, fail_on_diff, verbose);

    // Make sure some data was collected
    ASSERT_TRUE(trk_stats_fw.n_tracks > 0u);
    ASSERT_TRUE(n_surfaces_fw.n_total() > 0u);

    ASSERT_TRUE(trk_stats_bw.n_tracks > 0u);
    ASSERT_TRUE(n_surfaces_bw.n_total() > 0u);

    // Check stats
    auto n_tracks{static_cast<double>(trk_stats_fw.n_tracks)};
    EXPECT_TRUE(static_cast<double>(trk_stats_fw.n_tracks_w_holes) / n_tracks <
                0.01);
    EXPECT_TRUE(static_cast<double>(trk_stats_fw.n_tracks_w_extra) / n_tracks <
                0.2);
    EXPECT_TRUE(trk_stats_fw.n_max_missed_per_trk < 2);

    n_tracks = static_cast<double>(trk_stats_bw.n_tracks);
    EXPECT_TRUE(static_cast<double>(trk_stats_bw.n_tracks_w_holes) / n_tracks <
                0.01);
    EXPECT_TRUE(static_cast<double>(trk_stats_bw.n_tracks_w_extra) / n_tracks <
                0.2);
    EXPECT_TRUE(trk_stats_bw.n_max_missed_per_trk < 2);
}
