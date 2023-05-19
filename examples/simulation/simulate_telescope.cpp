/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
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

// detray include(s).
#include "detray/detectors/create_telescope_detector.hpp"
#include "detray/masks/unbounded.hpp"
#include "detray/simulation/event_generator/track_generators.hpp"
#include "detray/simulation/simulator.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Boost include(s).
#include <boost/filesystem.hpp>

using namespace traccc;
namespace po = boost::program_options;

int simulate(std::string output_directory, unsigned int events,
             const traccc::particle_gen_options<scalar>& pg_opts) {

    // Use deterministic random number generator for testing
    using uniform_gen_t =
        detray::random_numbers<scalar, std::uniform_real_distribution<scalar>,
                               std::seed_seq>;

    // Memory resource
    vecmem::host_memory_resource host_mr;

    /*****************************
     * Build a telescope geometry
     *****************************/

    // Plane alignment direction (aligned to x-axis)
    detray::detail::ray<transform3> traj{{0, 0, 0}, 0, {1, 0, 0}, -1};
    // Position of planes (in mm unit)
    std::vector<scalar> plane_positions = {20.,  40., 60., 80., 100.,
                                           120., 140, 160, 180.};

    // Detector type
    using detector_type =
        detray::detector<detray::detector_registry::template telescope_detector<
                             detray::unbounded<detray::rectangle2D<>>>,
                         covfie::field>;

    // B field value and its type
    const vector3 B{2 * detray::unit<scalar>::T, 0, 0};
    using b_field_t = typename detector_type::bfield_type;

    // Create the detector
    const auto mat = detray::silicon_tml<scalar>();
    const scalar thickness = 0.5 * detray::unit<scalar>::mm;

    // Use rectangle surfaces
    detray::mask<detray::unbounded<detray::rectangle2D<>>> rectangle{
        0u, 10000.f * detray::unit<scalar>::mm,
        10000.f * detray::unit<scalar>::mm};

    const detector_type det = create_telescope_detector(
        host_mr,
        b_field_t(b_field_t::backend_t::configuration_t{B[0], B[1], B[2]}),
        rectangle, plane_positions, mat, thickness, traj);

    /***************************
     * Generate simulation data
     ***************************/

    // Origin of particles
    auto generator =
        detray::random_track_generator<traccc::free_track_parameters,
                                       uniform_gen_t>(
            pg_opts.gen_nparticles, pg_opts.vertex, pg_opts.vertex_stddev,
            pg_opts.mom_range, pg_opts.theta_range, pg_opts.phi_range);

    // Smearing value for measurements
    detray::measurement_smearer<transform3> meas_smearer(
        50 * detray::unit<scalar>::um, 50 * detray::unit<scalar>::um);

    // Run simulator
    const std::string full_path = io::data_directory() + output_directory;

    boost::filesystem::create_directories(full_path);

    auto sim = detray::simulator(events, det, std::move(generator),
                                 meas_smearer, full_path);
    sim.run();

    return 1;
}

// The main routine
//
int main(int argc, char* argv[]) {
    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    desc.add_options()("output_directory", po::value<std::string>()->required(),
                       "specify the directory of output data");
    desc.add_options()("events", po::value<unsigned int>()->required(),
                       "number of events");
    traccc::particle_gen_options<scalar> pg_opts(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    auto output_directory = vm["output_directory"].as<std::string>();
    auto events = vm["events"].as<unsigned int>();
    pg_opts.read(vm);

    std::cout << "Running " << argv[0] << " " << output_directory << " "
              << events << std::endl;

    return simulate(output_directory, events, pg_opts);
}
