/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/options.hpp"
#include "traccc/options/particle_gen_options.hpp"
#include "traccc/options/propagation_options.hpp"
#include "traccc/options/telescope_detector_options.hpp"
#include "traccc/simulation/measurement_smearer.hpp"
#include "traccc/simulation/simulator.hpp"
#include "traccc/simulation/smearing_writer.hpp"

// detray include(s).
#include "detray/detectors/bfield.hpp"
#include "detray/detectors/create_telescope_detector.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes/rectangle2D.hpp"
#include "detray/io/frontend/detector_writer.hpp"
#include "detray/materials/material.hpp"
#include "detray/navigation/detail/ray.hpp"
#include "detray/simulation/event_generator/track_generators.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Boost include(s).
#include <boost/filesystem.hpp>

using namespace traccc;
namespace po = boost::program_options;

int simulate(std::string output_directory, unsigned int events,
             const traccc::particle_gen_options<scalar>& pg_opts,
             const traccc::propagation_options<scalar>& propagation_opts,
             const traccc::telescope_detector_options<scalar>& telescope_opts) {

    // Use deterministic random number generator for testing
    using uniform_gen_t =
        detray::random_numbers<scalar, std::uniform_real_distribution<scalar>,
                               std::seed_seq>;

    // Memory resource
    vecmem::host_memory_resource host_mr;

    /*****************************
     * Build a telescope geometry
     *****************************/

    // Plane alignment direction (aligned to z-axis)
    detray::detail::ray<transform3> traj{{0, 0, 0}, 0, {0, 0, 1}, -1};
    // Position of planes (in mm unit)
    std::vector<scalar> plane_positions;

    for (unsigned int i = 0; i < telescope_opts.n_planes; i++) {
        plane_positions.push_back((i + 1) * telescope_opts.spacing);
    }

    // B field value and its type
    using b_field_t = covfie::field<detray::bfield::const_bknd_t>;
    const vector3 B{0, 0, 2 * detray::unit<scalar>::T};
    auto field = detray::bfield::create_const_field(B);

    // Set material and thickness
    detray::material<scalar> mat;
    if (!telescope_opts.empty_material) {
        mat = detray::silicon<scalar>();
    } else {
        mat = detray::vacuum<scalar>();
    }

    const scalar thickness = telescope_opts.thickness;

    // Use rectangle surfaces
    detray::mask<detray::rectangle2D> rectangle{0u, telescope_opts.half_length,
                                                telescope_opts.half_length};

    detray::tel_det_config<> tel_cfg{rectangle};
    tel_cfg.positions(plane_positions);
    tel_cfg.module_material(mat);
    tel_cfg.mat_thickness(thickness);
    tel_cfg.pilot_track(traj);

    const auto [det, name_map] = create_telescope_detector(host_mr, tel_cfg);

    /***************************
     * Generate simulation data
     ***************************/

    // Origin of particles
    using generator_type =
        detray::random_track_generator<traccc::free_track_parameters,
                                       uniform_gen_t>;
    generator_type::configuration gen_cfg{};
    gen_cfg.n_tracks(pg_opts.gen_nparticles);
    gen_cfg.origin(pg_opts.vertex);
    gen_cfg.origin_stddev(pg_opts.vertex_stddev);
    gen_cfg.phi_range(pg_opts.phi_range[0], pg_opts.phi_range[1]);
    gen_cfg.theta_range(pg_opts.theta_range[0], pg_opts.theta_range[1]);
    gen_cfg.mom_range(pg_opts.mom_range[0], pg_opts.mom_range[1]);
    gen_cfg.charge(pg_opts.charge);
    generator_type generator(gen_cfg);

    // Smearing value for measurements
    traccc::measurement_smearer<transform3> meas_smearer(
        telescope_opts.smearing, telescope_opts.smearing);

    // Type declarations
    using detector_type = decltype(det);
    using generator_type = decltype(generator);
    using writer_type =
        traccc::smearing_writer<traccc::measurement_smearer<transform3>>;

    // Writer config
    typename writer_type::config smearer_writer_cfg{meas_smearer};

    // Run simulator
    const std::string full_path = io::data_directory() + output_directory;

    boost::filesystem::create_directories(full_path);

    auto sim = traccc::simulator<detector_type, b_field_t, generator_type,
                                 writer_type>(
        events, det, field, std::move(generator), std::move(smearer_writer_cfg),
        full_path);
    sim.get_config().propagation = propagation_opts.propagation;

    sim.run();

    // Create detector file
    auto writer_cfg = detray::io::detector_writer_config{}
                          .format(detray::io::format::json)
                          .replace_files(true);
    detray::io::write_detector(det, name_map, writer_cfg);

    return 1;
}

// The main routine
//
int main(int argc, char* argv[]) {
    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    desc.add_options()("output-directory", po::value<std::string>()->required(),
                       "specify the directory of output data");
    desc.add_options()("events", po::value<unsigned int>()->required(),
                       "number of events");
    traccc::particle_gen_options<scalar> pg_opts(desc);
    traccc::propagation_options<scalar> propagation_opts(desc);
    traccc::telescope_detector_options<scalar> telescope_opts(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    auto output_directory = vm["output-directory"].as<std::string>();
    auto events = vm["events"].as<unsigned int>();
    pg_opts.read(vm);
    propagation_opts.read(vm);
    telescope_opts.read(vm);

    std::cout << "Running " << argv[0] << " " << output_directory << " "
              << events << std::endl;

    return simulate(output_directory, events, pg_opts, propagation_opts,
                    telescope_opts);
}
