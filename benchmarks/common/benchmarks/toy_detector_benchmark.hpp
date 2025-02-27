/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Traccc include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"
#include "traccc/simulation/measurement_smearer.hpp"
#include "traccc/simulation/simulator.hpp"
#include "traccc/simulation/smearing_writer.hpp"

// Detray include(s).
#include <detray/detectors/bfield.hpp>
#include <detray/io/frontend/detector_writer.hpp>
#include <detray/test/utils/detectors/build_toy_detector.hpp>
#include <detray/test/utils/simulation/event_generator/track_generators.hpp>
#include <detray/tracks/ray.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Boost include(s).
#include <boost/filesystem.hpp>

// Google Benchmark include(s).
#include <benchmark/benchmark.h>

class ToyDetectorBenchmark : public benchmark::Fixture {
    public:
    // VecMem memory resource(s)
    vecmem::host_memory_resource host_mr;

    static const int n_events = 100u;
    static const int n_tracks = 5000u;

    std::vector<traccc::edm::spacepoint_collection::host> spacepoints;
    std::vector<traccc::measurement_collection_types::host> measurements;

    // Configs
    traccc::seedfinder_config seeding_cfg;
    traccc::seedfilter_config filter_cfg;
    traccc::spacepoint_grid_config grid_cfg{seeding_cfg};
    traccc::finding_config finding_cfg;
    traccc::fitting_config fitting_cfg;

    static constexpr std::array<float, 2> phi_range{
        -traccc::constant<float>::pi, traccc::constant<float>::pi};
    static constexpr std::array<float, 2> eta_range{-3, 3};
    static constexpr std::array<float, 2> mom_range{
        10.f * traccc::unit<float>::GeV, 100.f * traccc::unit<float>::GeV};

    static inline const std::string sim_dir = "toy_detector_benchmark/";

    // Detector type
    using detector_type = traccc::toy_detector::host;
    using algebra_type = typename detector_type::algebra_type;
    using scalar_type = detector_type::scalar_type;

    // B field value and its type
    // @TODO: Set B field as argument
    using b_field_t = covfie::field<detray::bfield::const_bknd_t<scalar_type>>;

    static constexpr traccc::vector3 B{0, 0, 2 * traccc::unit<scalar_type>::T};

    ToyDetectorBenchmark() {

        std::cout << "Please be patient. It may take some time to generate "
                     "the simulation data."
                  << std::endl;

        // Apply correct propagation config
        apply_propagation_config(finding_cfg.propagation);
        apply_propagation_config(fitting_cfg.propagation);

        // Use deterministic random number generator for testing
        using uniform_gen_t = detray::detail::random_numbers<
            scalar_type, std::uniform_real_distribution<scalar_type>>;

        // Build the detector
        auto [det, name_map] =
            detray::build_toy_detector<algebra_type>(host_mr, get_toy_config());

        // B field
        auto field = detray::bfield::create_const_field<scalar_type>(B);

        // Origin of particles
        using generator_type = detray::random_track_generator<
            traccc::free_track_parameters<algebra_type>, uniform_gen_t>;
        generator_type::configuration gen_cfg{};
        gen_cfg.n_tracks(n_tracks);
        gen_cfg.phi_range(phi_range);
        gen_cfg.eta_range(eta_range);
        gen_cfg.mom_range(mom_range);
        generator_type generator(gen_cfg);

        // Smearing value for measurements
        traccc::measurement_smearer<traccc::default_algebra> meas_smearer(
            50 * traccc::unit<scalar_type>::um,
            50 * traccc::unit<scalar_type>::um);

        // Type declarations
        using writer_type =
            traccc::smearing_writer<traccc::measurement_smearer<algebra_type>>;

        // Writer config
        typename writer_type::config smearer_writer_cfg{meas_smearer};

        // Run simulator
        const std::string full_path = traccc::io::data_directory() + sim_dir;

        boost::filesystem::create_directories(full_path);

        auto sim = traccc::simulator<detector_type, b_field_t, generator_type,
                                     writer_type>(
            detray::muon<scalar_type>(), n_events, det, field,
            std::move(generator), std::move(smearer_writer_cfg), full_path);

        // Same propagation configuration for sim and reco
        apply_propagation_config(sim.get_config().propagation);
        // Set constrained step size to 1 mm
        sim.get_config().propagation.stepping.step_constraint =
            1.f * traccc::unit<float>::mm;

        sim.run();

        // Write detector file
        auto writer_cfg = detray::io::detector_writer_config{}
                              .format(detray::io::format::json)
                              .replace_files(true)
                              .write_grids(true)
                              .write_material(true)
                              .path(full_path);
        detray::io::write_detector(det, name_map, writer_cfg);
    }

    detray::toy_det_config<scalar_type> get_toy_config() const {

        // Create the toy geometry
        detray::toy_det_config<scalar_type> toy_cfg{};
        toy_cfg.n_brl_layers(4u).n_edc_layers(7u).do_check(false);

        // @TODO: Increase the material budget again
        toy_cfg.module_mat_thickness(0.11f * traccc::unit<scalar_type>::mm);

        return toy_cfg;
    }

    void apply_propagation_config(detray::propagation::config& cfg) const {
        // Configure the propagation for the toy detector
        // cfg.navigation.search_window = {3, 3};
        cfg.navigation.overstep_tolerance = -300.f * traccc::unit<float>::um;
        cfg.navigation.min_mask_tolerance = 1e-5f * traccc::unit<float>::mm;
        cfg.navigation.max_mask_tolerance = 3.f * traccc::unit<float>::mm;
        cfg.navigation.mask_tolerance_scalor = 0.05f;
    }

    void SetUp(::benchmark::State& /*state*/) {

        // Read events
        for (std::size_t i_evt = 0; i_evt < n_events; i_evt++) {

            // Read the hits from the relevant event file
            traccc::edm::spacepoint_collection::host sp{host_mr};
            traccc::measurement_collection_types::host meas{&host_mr};
            traccc::io::read_spacepoints(sp, meas, i_evt, sim_dir);
            spacepoints.push_back(sp);
            measurements.push_back(meas);
        }
    }
};
