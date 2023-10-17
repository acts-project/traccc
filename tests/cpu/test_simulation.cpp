/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/csv/make_hit_reader.hpp"
#include "traccc/io/csv/make_measurement_hit_id_reader.hpp"
#include "traccc/io/csv/make_measurement_reader.hpp"
#include "traccc/io/csv/make_particle_reader.hpp"
#include "traccc/simulation/simulator.hpp"

// Detray include(s).
#include "detray/detectors/create_telescope_detector.hpp"
#include "detray/detectors/create_toy_geometry.hpp"
#include "detray/geometry/surface.hpp"
#include "detray/io/common/detail/utils.hpp"
#include "detray/masks/masks.hpp"
#include "detray/masks/unbounded.hpp"
#include "detray/simulation/event_generator/track_generators.hpp"
#include "detray/tracks/bound_track_parameters.hpp"
#include "detray/utils/statistics.hpp"

// GTest include(s).
#include <gtest/gtest.h>

using namespace traccc;
using namespace detray;

constexpr scalar tol{1e-7f};

TEST(simulation, simulation) {

    const mask<line<false, line_intersector>> ln{
        0u, 10.f * detray::unit<scalar>::mm, 50.f * detray::unit<scalar>::mm};

    const mask<rectangle2D<plane_intersector>> re{
        0u, 10.f * detray::unit<scalar>::mm, 10.f * detray::unit<scalar>::mm};

    bound_track_parameters<transform3> bound_params;
    auto& bound_vec = bound_params.vector();
    getter::element(bound_vec, traccc::e_bound_loc0, 0u) = 1.f;
    getter::element(bound_vec, traccc::e_bound_loc1, 0u) = 2.f;

    measurement_smearer<transform3> smearer(0.f, 0.f);

    io::csv::measurement iomeas1;
    smearer(ln, {-3.f, 2.f}, bound_params, iomeas1);
    ASSERT_NEAR(iomeas1.local0, 0.f, tol);
    ASSERT_NEAR(iomeas1.local1, 0.f, tol);

    io::csv::measurement iomeas2;
    smearer(ln, {2.f, -5.f}, bound_params, iomeas2);
    ASSERT_NEAR(iomeas2.local0, 3.f, tol);
    ASSERT_NEAR(iomeas2.local1, 0.f, tol);

    io::csv::measurement iomeas3;
    smearer(re, {2.f, -5.f}, bound_params, iomeas3);
    ASSERT_NEAR(iomeas3.local0, 3.f, tol);
    ASSERT_NEAR(iomeas3.local1, -3.f, tol);
}

GTEST_TEST(detray_simulation, toy_geometry_simulation) {

    // Use deterministic random number generator for testing
    using normal_gen_t =
        random_numbers<scalar, std::normal_distribution<scalar>, std::seed_seq>;

    // Create geometry
    vecmem::host_memory_resource host_mr;

    // Create B field
    const vector3 B{0.f, 0.f, 2.f * detray::unit<scalar>::T};

    // Create geometry
    using b_field_t = decltype(create_toy_geometry(host_mr).first)::bfield_type;
    const auto [detector, names] = create_toy_geometry(
        host_mr,
        b_field_t(b_field_t::backend_t::configuration_t{B[0], B[1], B[2]}));

    using geo_cxt_t = typename decltype(detector)::geometry_context;
    const geo_cxt_t ctx{};

    // Create track generator
    constexpr unsigned int n_tracks{2500u};
    const vector3 ori{0.f, 0.f, 0.f};
    auto generator =
        random_track_generator<free_track_parameters<transform3>, normal_gen_t>(
            n_tracks, ori);

    // Create smearer
    measurement_smearer<transform3> smearer(67.f * detray::unit<scalar>::um,
                                            170.f * detray::unit<scalar>::um);

    std::size_t n_events{10u};

    using detector_type = decltype(detector);
    using generator_type = decltype(generator);
    using writer_type = smearing_writer<measurement_smearer<transform3>>;

    typename writer_type::config writer_cfg{smearer};

    auto sim = simulator<detector_type, generator_type, writer_type>(
        n_events, detector, std::move(generator), std::move(writer_cfg));

    // Lift step size constraints
    sim.get_config().step_constraint = std::numeric_limits<scalar>::max();

    // Do the simulation
    sim.run();

    for (std::size_t i_event = 0u; i_event < n_events; i_event++) {

        std::vector<io::csv::particle> particles;
        auto particle_reader = io::csv::make_particle_reader(
            detail::get_event_filename(i_event, "-particles.csv"));
        io::csv::particle io_particle;
        while (particle_reader.read(io_particle)) {
            particles.push_back(io_particle);
        }

        std::vector<io::csv::hit> hits;
        auto hit_reader = io::csv::make_hit_reader(
            detail::get_event_filename(i_event, "-hits.csv"));
        io::csv::hit io_hit;
        while (hit_reader.read(io_hit)) {
            hits.push_back(io_hit);
        }

        std::vector<io::csv::measurement> measurements;
        auto measurement_reader = io::csv::make_measurement_reader(
            detail::get_event_filename(i_event, "-measurements.csv"));
        io::csv::measurement io_measurement;
        while (measurement_reader.read(io_measurement)) {
            measurements.push_back(io_measurement);
        }

        std::vector<io::csv::measurement_hit_id> meas_hit_ids;
        auto measurement_hit_id_reader =
            io::csv::make_measurement_hit_id_reader(detail::get_event_filename(
                i_event, "-measurement-simhit-map.csv"));
        io::csv::measurement_hit_id io_meas_hit_id;
        while (measurement_hit_id_reader.read(io_meas_hit_id)) {
            meas_hit_ids.push_back(io_meas_hit_id);
        }

        ASSERT_EQ(particles.size(), n_tracks);
        ASSERT_TRUE(not measurements.empty());
        ASSERT_EQ(hits.size(), measurements.size());
        ASSERT_EQ(hits.size(), meas_hit_ids.size());

        // Let's check if measurement smearing works correctly...
        std::vector<scalar> local0_diff;
        std::vector<scalar> local1_diff;

        const std::size_t nhits = hits.size();
        for (std::size_t i = 0u; i < nhits; i++) {
            const point3 pos{hits[i].tx, hits[i].ty, hits[i].tz};
            const vector3 mom{hits[i].tpx, hits[i].tpy, hits[i].tpz};
            const auto truth_local =
                surface{detector, geometry::barcode(hits[i].geometry_id)}
                    .global_to_local(ctx, pos, vector::normalize(mom));

            local0_diff.push_back(truth_local[0] - measurements[i].local0);
            local1_diff.push_back(truth_local[1] - measurements[i].local1);

            ASSERT_EQ(meas_hit_ids[i].hit_id, i);
            ASSERT_EQ(meas_hit_ids[i].measurement_id, i);
        }

        const auto var0 = statistics::variance(local0_diff);
        const auto var1 = statistics::variance(local1_diff);

        EXPECT_NEAR((std::sqrt(var0) - smearer.stddev[0]) / smearer.stddev[0],
                    0.f, 0.1f);
        EXPECT_NEAR((std::sqrt(var1) - smearer.stddev[1]) / smearer.stddev[1],
                    0.f, 0.1f);
    }
}

// Test parameters: <initial momentum, theta direction>
class TelescopeDetectorSimulation
    : public ::testing::TestWithParam<std::tuple<scalar, scalar>> {};

TEST_P(TelescopeDetectorSimulation, telescope_detector_simulation) {

    // Create geometry
    vecmem::host_memory_resource host_mr;

    // Build from given module positions
    std::vector<scalar> positions = {0.f,   50.f,  100.f, 150.f, 200.f, 250.f,
                                     300.f, 350.f, 400.f, 450.f, 500.f};

    // A thickness larger than 0.1 cm will flip the track direction of low
    // energy (or non-relativistic) particle due to the large scattering
    const scalar thickness = 0.005f * unit<scalar>::cm;

    tel_det_config<rectangle2D<>> tel_cfg{1000.f * unit<scalar>::mm,
                                          1000.f * unit<scalar>::mm};
    tel_cfg.positions(positions).mat_thickness(thickness);

    const auto [detector, names] = create_telescope_detector(host_mr, tel_cfg);

    // Momentum
    const scalar mom = std::get<0>(GetParam());

    // Create track generator
    constexpr unsigned int theta_steps{1u};
    constexpr unsigned int phi_steps{1u};
    const vector3 ori{0.f, 0.f, 0.f};
    const scalar theta = std::get<1>(GetParam());
    auto generator = uniform_track_generator<free_track_parameters<transform3>>(
        theta_steps, phi_steps, ori, mom, {theta, theta}, {0.f, 0.f});

    // Create smearer
    measurement_smearer<transform3> smearer(50.f * unit<scalar>::um,
                                            50.f * unit<scalar>::um);

    std::size_t n_events{1000u};

    using detector_type = decltype(detector);
    using generator_type = decltype(generator);
    using writer_type = smearing_writer<measurement_smearer<transform3>>;

    typename writer_type::config writer_cfg{smearer};

    auto sim = simulator<detector_type, generator_type, writer_type>(
        n_events, detector, std::move(generator), std::move(writer_cfg));

    // Lift step size constraints
    sim.get_config().step_constraint = std::numeric_limits<scalar>::max();

    // Run simulation
    sim.run();

    for (std::size_t i_event{0u}; i_event < n_events; i_event++) {

        std::vector<io::csv::measurement> measurements;
        auto measurement_reader = io::csv::make_measurement_reader(
            detail::get_event_filename(i_event, "-measurements.csv"));
        io::csv::measurement io_measurement;
        while (measurement_reader.read(io_measurement)) {
            measurements.push_back(io_measurement);
        }

        // Make sure that number of measurements is equal to the number of
        // physical planes
        ASSERT_EQ(measurements.size(), positions.size());
    }
}

INSTANTIATE_TEST_SUITE_P(Simulation1, TelescopeDetectorSimulation,
                         ::testing::Values(std::make_tuple(
                             0.1f * detray::unit<scalar>::GeV, 0.01f)));

INSTANTIATE_TEST_SUITE_P(
    Simulation2, TelescopeDetectorSimulation,
    ::testing::Values(std::make_tuple(1.f * detray::unit<scalar>::GeV, 0.01f)));

INSTANTIATE_TEST_SUITE_P(Simulation3, TelescopeDetectorSimulation,
                         ::testing::Values(std::make_tuple(
                             10.f * detray::unit<scalar>::GeV, 0.01f)));

INSTANTIATE_TEST_SUITE_P(Simulation4, TelescopeDetectorSimulation,
                         ::testing::Values(std::make_tuple(
                             100.f * detray::unit<scalar>::GeV, 0.01f)));

INSTANTIATE_TEST_SUITE_P(Simulation5, TelescopeDetectorSimulation,
                         ::testing::Values(std::make_tuple(
                             0.1f * detray::unit<scalar>::GeV, 0.01f)));

INSTANTIATE_TEST_SUITE_P(
    Simulation6, TelescopeDetectorSimulation,
    ::testing::Values(std::make_tuple(1.f * detray::unit<scalar>::GeV, 0.01f)));

INSTANTIATE_TEST_SUITE_P(Simulation7, TelescopeDetectorSimulation,
                         ::testing::Values(std::make_tuple(
                             10.f * detray::unit<scalar>::GeV, 0.01f)));

INSTANTIATE_TEST_SUITE_P(Simulation8, TelescopeDetectorSimulation,
                         ::testing::Values(std::make_tuple(
                             100.f * detray::unit<scalar>::GeV, 0.01f)));

INSTANTIATE_TEST_SUITE_P(
    Simulation9, TelescopeDetectorSimulation,
    ::testing::Values(std::make_tuple(0.1f * detray::unit<scalar>::GeV,
                                      detray::constant<scalar>::pi / 8.f)));

INSTANTIATE_TEST_SUITE_P(
    Simulation10, TelescopeDetectorSimulation,
    ::testing::Values(std::make_tuple(1.f * detray::unit<scalar>::GeV,
                                      detray::constant<scalar>::pi / 8.f)));

INSTANTIATE_TEST_SUITE_P(
    Simulation11, TelescopeDetectorSimulation,
    ::testing::Values(std::make_tuple(10.f * detray::unit<scalar>::GeV,
                                      detray::constant<scalar>::pi / 8.f)));

INSTANTIATE_TEST_SUITE_P(
    Simulation12, TelescopeDetectorSimulation,
    ::testing::Values(std::make_tuple(100.f * detray::unit<scalar>::GeV,
                                      detray::constant<scalar>::pi / 8.f)));

INSTANTIATE_TEST_SUITE_P(
    Simulation13, TelescopeDetectorSimulation,
    ::testing::Values(std::make_tuple(0.1f * detray::unit<scalar>::GeV,
                                      detray::constant<scalar>::pi / 6.f)));

INSTANTIATE_TEST_SUITE_P(
    Simulation14, TelescopeDetectorSimulation,
    ::testing::Values(std::make_tuple(1.f * detray::unit<scalar>::GeV,
                                      detray::constant<scalar>::pi / 6.f)));

INSTANTIATE_TEST_SUITE_P(
    Simulation15, TelescopeDetectorSimulation,
    ::testing::Values(std::make_tuple(10.f * detray::unit<scalar>::GeV,
                                      detray::constant<scalar>::pi / 6.f)));

INSTANTIATE_TEST_SUITE_P(
    Simulation16, TelescopeDetectorSimulation,
    ::testing::Values(std::make_tuple(100.f * detray::unit<scalar>::GeV,
                                      detray::constant<scalar>::pi / 6.f)));