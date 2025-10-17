/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/bfield/construct_const_bfield.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/performance/kalman_filter_comparison.hpp"
#include "traccc/simulation/event_generators.hpp"
#include "traccc/simulation/simulator.hpp"

// Test include(s).
#include "tests/test_detectors.hpp"
#include "tests/toy_detector_fixture.hpp"

// Detray include(s).
#include <detray/io/frontend/detector_reader.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s)
#include <filesystem>

using namespace traccc;

constexpr std::size_t n_events{10u};
constexpr std::size_t n_tracks{1000u};

constexpr detray::pdg_particle ptc_type{detray::muon<scalar>()};

// Truth particle cuts
constexpr scalar min_p{50.f * traccc::unit<traccc::scalar>::MeV};
constexpr scalar max_r{75.f * traccc::unit<traccc::scalar>::mm};

/// Test suite for navigation tests for the CKF
class KF_intergration_test_toy_detector
    : public ToyDetectorFixture,
      public ::testing::WithParamInterface<std::tuple<
          float, float, float, float, std::size_t, bool, bool, bool>> {};

/// Test the detray navigation on simulated tracks
TEST_P(KF_intergration_test_toy_detector, toy_detector) {

    using detector_t = traccc::default_detector::host;
    using algebra_t = typename detector_t::algebra_type;
    using b_field_t = covfie::field<traccc::const_bfield_backend_t<scalar>>;
    using track_t = traccc::free_track_parameters<algebra_t>;

    using generator_t = detray::random_track_generator<track_t>;
    using writer_t = smearing_writer<measurement_smearer<algebra_t>>;
    using simulator_t = simulator<detector_t, b_field_t, generator_t, writer_t>;

    vecmem::host_memory_resource host_mr;

    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        "KF_intergration_test_toy_detector_toy_detector",
        traccc::Logging::Level::INFO);

    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file("toy_detector_geometry.json")
        .add_file("toy_detector_surface_grids.json")
        .do_check(true);
    if (std::get<5>(GetParam())) {
        reader_cfg.add_file("toy_detector_material_maps.json");
        // reader_cfg.add_file("toy_detector_homogeneous_material.json");
    }

    auto [io_det, names] =
        detray::io::read_detector<traccc::default_detector::host>(host_mr,
                                                                  reader_cfg);
    traccc::host_detector host_det{};
    host_det.template set<detector_traits<typename detector_t::metadata>>(
        std::move(io_det));
    const auto& det = host_det.template as<traccc::detector_traits<typename detector_t::metadata>>();

    // Create B field
    b_field_t field = traccc::construct_const_bfield(B)
                          .as_field<traccc::const_bfield_backend_t<scalar>>();

    // Create track generator
    const scalar pT{std::get<0>(GetParam())};
    generator_t::configuration gen_cfg{};
    gen_cfg.n_tracks(n_tracks).eta_range(-3, 3).p_T(pT).randomize_charge(true);
    // Choose different random seed than detray for more test coverage
    gen_cfg.seed(135346);

    // Create data directory
    std::filesystem::path data_dir{traccc::io::data_directory()};
    std::filesystem::path outdir{"fast_track_simulation/toy_detector_pT_" +
                                 std::to_string(pT) + "_GeV"};

    std::filesystem::path full_path = data_dir / outdir;
    if (!std::filesystem::exists(full_path)) {
        if (std::error_code err;
            !std::filesystem::create_directories(full_path, err)) {
            throw std::runtime_error(err.message());
        }
    }

    // Create measurement smearer
    measurement_smearer<algebra_t> smearer(smearing[0], smearing[1]);
    auto sim = simulator_t(ptc_type, n_events, det, field, generator_t{gen_cfg},
                           writer_t::config{smearer}, full_path.string());

    // Propagation config for the simulation
    detray::propagation::config prop_cfg{};
    prop_cfg.navigation.search_window = search_window;  //< toy detector grids

    sim.get_config().propagation = prop_cfg;
    sim.get_config().propagation.stepping.step_constraint = step_constraint;
    sim.get_config().do_multiple_scattering = std::get<6>(GetParam());
    sim.get_config().do_energy_loss = std::get<7>(GetParam());
    sim.get_config().min_pT(10.f * traccc::unit<scalar>::MeV);

    // Run the simulation: Produces data files
    sim.run();

    // Specific config for the navigation test
    prop_cfg.navigation.n_scattering_stddev = 2;
    prop_cfg.navigation.accumulated_error = 0.f;
    prop_cfg.navigation.estimate_scattering_noise = true;
    //prop_cfg.navigation.min_mask_tolerance = 0.15f;
    // Try with a flat tolerance instead
    if (!prop_cfg.navigation.estimate_scattering_noise) {
        prop_cfg.navigation.min_mask_tolerance = std::get<1>(GetParam());
    }
    //prop_cfg.navigation.mask_tolerance_scalor = 1.f;
    prop_cfg.navigation.overstep_tolerance = -1000.f * traccc::unit<float>::um;
    prop_cfg.navigation.max_mask_tolerance =
        std::get<1>(GetParam()) + 3.f * traccc::unit<float>::mm;

    const bool success = kalman_filter_comparison(
        &host_det, names, prop_cfg, outdir, n_events, logger->clone(),
        std::get<6>(GetParam()), std::get<7>(GetParam()), false, ptc_type,
        stddevs, B, min_p, max_r);

    ASSERT_TRUE(success);
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
/*INSTANTIATE_TEST_SUITE_P(
    pT_100GeV_no_mat, KF_intergration_test_toy_detector,
    ::testing::Values(std::make_tuple(100.f * traccc::unit<scalar>::GeV,
                                      1e-5f * traccc::unit<float>::mm, 0.001f,
                                      0.001f, 1u, false, false, false)));

INSTANTIATE_TEST_SUITE_P(
    pT_10GeV_no_mat, KF_intergration_test_toy_detector,
    ::testing::Values(std::make_tuple(10.f * traccc::unit<scalar>::GeV,
                                      1e-5f * traccc::unit<float>::mm, 0.001f,
                                      0.001f, 1u, false, false, false)));*/

INSTANTIATE_TEST_SUITE_P(
    pT_05GeV_no_mat, KF_intergration_test_toy_detector,
    ::testing::Values(std::make_tuple(0.5f * traccc::unit<scalar>::GeV,
                                      1e-5f * traccc::unit<float>::mm, 0.001f,
                                      0.001f, 1u, false, false, false)));

// No scattering - navigation should work (material interactor models e-loss)
/*INSTANTIATE_TEST_SUITE_P(
    pT_100GeV_only_eloss, KF_intergration_test_toy_detector,
    ::testing::Values(std::make_tuple(100.f * traccc::unit<scalar>::GeV,
                                      1e-3f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, false, true)));
INSTANTIATE_TEST_SUITE_P(
    pT_10GeV_only_eloss, KF_intergration_test_toy_detector,
    ::testing::Values(std::make_tuple(10.f * traccc::unit<scalar>::GeV,
                                      1e-3f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, false, true)));*/

INSTANTIATE_TEST_SUITE_P(
    pT_05GeV_only_eloss, KF_intergration_test_toy_detector,
    ::testing::Values(std::make_tuple(0.5f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.005f,
                                      0.005f, 4u, true, false, true)));

// No energy loss - navigation has to compensate the scattering angle
// (turn off the material in the detector to prevent bethe-bloch corrections)
/*INSTANTIATE_TEST_SUITE_P(
    pT_100GeV_only_scatt, KF_intergration_test_toy_detector,
    ::testing::Values(std::make_tuple(100.f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, true, false)));

INSTANTIATE_TEST_SUITE_P(
    pT_10GeV_only_scatt, KF_intergration_test_toy_detector,
    ::testing::Values(std::make_tuple(10.f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.001f,
                                      0.005f, 4u, true, true, false)));
INSTANTIATE_TEST_SUITE_P(
    pT_05GeV_only_scatt, KF_intergration_test_toy_detector,
    ::testing::Values(std::make_tuple(0.5f * traccc::unit<scalar>::GeV,
                                      1e-2f * traccc::unit<float>::mm, 0.005f,
                                      0.005f, 4u, true, true, false)));*/

// Nominal (e-loss + scattering)
INSTANTIATE_TEST_SUITE_P(
    pT_100GeV_nominal, KF_intergration_test_toy_detector,
    ::testing::Values(std::make_tuple(100.f * traccc::unit<scalar>::GeV,
                                      1.5f * traccc::unit<float>::mm, 0.01f,
                                      0.15f, 1u, true, true, true)));
INSTANTIATE_TEST_SUITE_P(
    pT_10GeV_nominal, KF_intergration_test_toy_detector,
    ::testing::Values(std::make_tuple(10.f * traccc::unit<scalar>::GeV,
                                      1.5f * traccc::unit<float>::mm, 0.01f,
                                      0.15f, 1u, true, true, true)));
INSTANTIATE_TEST_SUITE_P(
    pT_1GeV_nominal, KF_intergration_test_toy_detector,
    ::testing::Values(std::make_tuple(1.f * traccc::unit<scalar>::GeV,
                                      1.5f * traccc::unit<float>::mm, 0.01f,
                                      0.15f, 1u, true, true, true)));

INSTANTIATE_TEST_SUITE_P(
    pT_05GeV_nominal, KF_intergration_test_toy_detector,
    ::testing::Values(std::make_tuple(0.5f * traccc::unit<scalar>::GeV,
                                      1.5f * traccc::unit<float>::mm, 0.01f,
                                      0.5f, 3u, true, true, true)));
