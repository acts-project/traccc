/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../common/make_magnetic_field.hpp"
#include "traccc/bfield/construct_const_bfield.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/fitting/kalman_filter/kalman_actor.hpp"
#include "traccc/geometry/host_detector.hpp"
#include "traccc/io/data_format.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/simulation/event_generators.hpp"
#include "traccc/simulation/measurement_smearer.hpp"
#include "traccc/simulation/simulator.hpp"
#include "traccc/simulation/smearing_writer.hpp"
#include "traccc/utils/event_data.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/propagation.hpp"

// Options
#include "traccc/options/detector.hpp"
#include "traccc/options/generation.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/magnetic_field.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_propagation.hpp"

// Performance include(s).
#include "traccc/performance/kalman_filter_comparison.hpp"

// Detray include(s)
#include <detray/io/frontend/detector_reader.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s)
#include <filesystem>

int main(int argc, char* argv[]) {

    using detector_t = traccc::default_detector::host;
    // using algebra_t = typename detector_t::algebra_type;
    using vector3_t = typename detector_t::vector3_type;

    std::unique_ptr<const traccc::Logger> ilogger = traccc::getDefaultLogger(
        "KalmanFilterValidationCPU", traccc::Logging::Level::INFO);
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::magnetic_field bfield_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::generation generation_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::program_options program_opts{
        "Kalman Filter validation on the Host",
        {detector_opts, input_opts, generation_opts, propagation_opts},
        argc,
        argv,
        logger().cloneWithSuffix("Options")};

    // Memory resource used by the application.
    vecmem::host_memory_resource host_mr;

    TRACCC_INFO("Reading detector from file");

    // Set up the detector reader configuration.
    detray::io::detector_reader_config reader_cfg;
    reader_cfg.add_file(
        traccc::io::get_absolute_path(detector_opts.detector_file));
    if (!detector_opts.material_file.empty()) {
        reader_cfg.add_file(
            traccc::io::get_absolute_path(detector_opts.material_file));
    }
    if (!detector_opts.grid_file.empty()) {
        reader_cfg.add_file(
            traccc::io::get_absolute_path(detector_opts.grid_file));
    }

    // Read the detector.
    auto [io_det, names] =
        detray::io::read_detector<detector_t>(host_mr, reader_cfg);

    traccc::host_detector host_det{};
    host_det
        .template set<traccc::detector_traits<typename detector_t::metadata>>(
            std::move(io_det));
    const auto& det = host_det.template as<
        traccc::detector_traits<typename detector_t::metadata>>();

    // Create B-field
    const auto field = traccc::details::make_magnetic_field(bfield_opts);

    TRACCC_INFO("Preparing input data");

    // Check input dir
    std::filesystem::path data_dir{traccc::io::data_directory()};
    std::filesystem::path input_dir = data_dir / input_opts.directory;

    // Data dir does not exist, create default directory
    if (!std::filesystem::exists(input_dir)) {

        TRACCC_INFO("Input directory "
                    << input_dir << " does not exist: Creating data path");

        /*if (std::error_code err;
            !std::filesystem::create_directories(input_dir, err)) {
            throw std::runtime_error(err.message());
        }*/
        return EXIT_FAILURE;
    }

    // No existing truth data: Run fast sim
    if (!std::filesystem::exists(input_dir) ||
        std::filesystem::is_empty(input_dir)) {

        TRACCC_INFO("Input data does not exist");
        return EXIT_FAILURE;

        /*TRACCC_INFO("Generating fast sim data for "
                    << input_opts.events << " events (" << input_dir << ")\n");

        // Create B field
        /// B field value and its type
        using b_field_t =
            covfie::field<traccc::const_bfield_backend_t<traccc::scalar>>;
        b_field_t field =
            traccc::construct_const_bfield(B)
                .as_field<traccc::const_bfield_backend_t<traccc::scalar>>();

        // Origin of particles
        using generator_t =
            detray::random_track_generator<traccc::free_track_parameters<>>;
        using writer_t =
            traccc::smearing_writer<traccc::measurement_smearer<algebra_t>>;
        using simulator_t =
            traccc::simulator<detector_t, b_field_t, generator_t, writer_t>;

        generator_t::configuration gen_cfg{};
        gen_cfg.n_tracks(generation_opts.gen_nparticles);
        gen_cfg.origin(traccc::point3{generation_opts.vertex[0],
                                      generation_opts.vertex[1],
                                      generation_opts.vertex[2]});
        gen_cfg.origin_stddev(traccc::point3{generation_opts.vertex_stddev[0],
                                             generation_opts.vertex_stddev[1],
                                             generation_opts.vertex_stddev[2]});
        gen_cfg.phi_range(generation_opts.phi_range);
        gen_cfg.theta_range(generation_opts.theta_range);
        gen_cfg.p_T(generation_opts.mom_range[0]);
        gen_cfg.charge(generation_opts.ptc_type.charge());
        generator_t generator(gen_cfg);

        // Create measurement smearer
        traccc::measurement_smearer<algebra_t> smearer(smearing[0],
                                                       smearing[1]);
        auto sim = simulator_t(generation_opts.ptc_type, generation_opts.events,
                               det, field, std::move(generator),
                               writer_t::config{smearer}, input_dir);

        // General propagation options set by user
        sim.get_config().propagation = propagation_opts;

        // Specific configurations for the simulation
        sim.get_config().propagation.stepping.step_constraint =
            1.f * traccc::unit<traccc::scalar>::mm;
        sim.get_config().do_multiple_scattering =
            generation_opts.do_multiple_scattering;
        sim.get_config().do_energy_loss = generation_opts.do_energy_loss;
        sim.get_config().min_pT(50.f * traccc::unit<traccc::scalar>::MeV);

        // Run the simulation: Produces data files
        sim.run();*/
    } else {
        TRACCC_INFO("Reading truth data in " << input_opts.directory << "\n");
    }

    // Run the application.
    const bool success = traccc::kalman_filter_comparison(
        &host_det, names, propagation_opts, input_opts.directory,
        static_cast<unsigned int>(input_opts.events), logger().clone(),
        generation_opts.do_multiple_scattering, generation_opts.do_energy_loss,
        input_opts.use_acts_geom_source, generation_opts.ptc_type, {}, B);

    if (!success) {
        TRACCC_ERROR("Validation failed for: " << det.name(names));
    }
}
