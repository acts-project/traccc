/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/generation.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/output_data.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/simulation/measurement_smearer.hpp"
#include "traccc/simulation/simulator.hpp"
#include "traccc/simulation/smearing_writer.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/core/detector_metadata.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"
#include "detray/simulation/event_generator/track_generators.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Boost include(s).
#include <boost/filesystem.hpp>

using namespace traccc;
namespace po = boost::program_options;

// The main routine
//
int main(int argc, char* argv[]) {

    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    traccc::opts::detector det_opts(desc);
    traccc::opts::generation generation_opts(desc);
    traccc::opts::output_data output_opts(desc);
    traccc::opts::track_propagation propagation_opts(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    /// Type declarations
    using host_detector_type = detray::detector<>;
    using uniform_gen_t =
        detray::random_numbers<scalar, std::uniform_real_distribution<scalar>,
                               std::seed_seq>;
    using generator_type =
        detray::random_track_generator<traccc::free_track_parameters,
                                       uniform_gen_t>;

    // Read options
    det_opts.read(vm);
    generation_opts.read(vm);
    output_opts.read(vm);
    propagation_opts.read(vm);

    std::cout << "\nRunning detector simulation\n\n"
              << det_opts << "\n"
              << generation_opts << "\n"
              << output_opts << "\n"
              << propagation_opts << std::endl;

    // B field value and its type
    // @TODO: Set B field as argument
    using b_field_t = covfie::field<detray::bfield::const_bknd_t>;
    const traccc::vector3 B{0, 0, 2 * detray::unit<traccc::scalar>::T};
    auto field = detray::bfield::create_const_field(B);

    // Read the detector
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(traccc::io::data_directory() + det_opts.detector_file);
    if (!det_opts.material_file.empty()) {
        reader_cfg.add_file(traccc::io::data_directory() +
                            det_opts.material_file);
    }
    if (!det_opts.grid_file.empty()) {
        reader_cfg.add_file(traccc::io::data_directory() + det_opts.grid_file);
    }

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    auto [host_det, names] =
        detray::io::read_detector<host_detector_type>(host_mr, reader_cfg);

    /***************************
     * Generate simulation data
     ***************************/

    // Origin of particles
    generator_type::configuration gen_cfg{};
    gen_cfg.n_tracks(generation_opts.gen_nparticles);
    gen_cfg.origin(generation_opts.vertex);
    gen_cfg.origin_stddev(generation_opts.vertex_stddev);
    gen_cfg.phi_range(generation_opts.phi_range[0],
                      generation_opts.phi_range[1]);
    gen_cfg.theta_range(generation_opts.theta_range[0],
                        generation_opts.theta_range[1]);
    gen_cfg.mom_range(generation_opts.mom_range[0],
                      generation_opts.mom_range[1]);
    gen_cfg.charge(generation_opts.charge);
    generator_type generator(gen_cfg);

    // Smearing value for measurements
    traccc::measurement_smearer<transform3> meas_smearer(
        50 * detray::unit<scalar>::um, 50 * detray::unit<scalar>::um);

    using writer_type =
        traccc::smearing_writer<traccc::measurement_smearer<transform3>>;

    // Writer config
    typename writer_type::config smearer_writer_cfg{meas_smearer};

    // Run simulator
    const std::string full_path = io::data_directory() + output_opts.directory;

    boost::filesystem::create_directories(full_path);

    auto sim = traccc::simulator<host_detector_type, b_field_t, generator_type,
                                 writer_type>(
        generation_opts.events, host_det, field, std::move(generator),
        std::move(smearer_writer_cfg), full_path);

    sim.get_config().propagation = propagation_opts.config;

    sim.run();

    return 1;
}
