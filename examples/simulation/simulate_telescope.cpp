/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"

// detray include(s).
#include "detray/detectors/create_telescope_detector.hpp"
#include "detray/simulation/simulator.hpp"
#include "detray/simulation/track_generators.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Boost include(s).
#include <boost/filesystem.hpp>

using namespace traccc;
namespace po = boost::program_options;

int simulate(std::string output_directory, unsigned int events, scalar p0,
             scalar phi0) {

    // Memory resource
    vecmem::host_memory_resource host_mr;

    /*****************************
     * Build a telescope geometry
     *****************************/

    // Plane alignment direction (aligned to x-axis)
    detray::detail::ray<transform3> traj{{0, 0, 0}, 0, {1, 0, 0}, -1};
    // Position of planes (in mm unit)
    std::vector<scalar> plane_positions = {-10., 20., 40., 60.,  80., 100.,
                                           120., 140, 160, 180., 200.};

    // Detector type
    using detector_type =
        detray::detector<detray::detector_registry::telescope_detector,
                         covfie::field>;

    // B field value and its type
    const vector3 B{2 * detray::unit<scalar>::T, 0, 0};
    using b_field_t = typename detector_type::bfield_type;

    // Create the detector
    const auto mat = detray::silicon_tml<scalar>();
    const scalar thickness = 0.5 * detray::unit<scalar>::mm;

    const detector_type det = create_telescope_detector(
        host_mr,
        b_field_t(b_field_t::backend_t::configuration_t{B[0], B[1], B[2]}),
        plane_positions, traj, 100000. * detray::unit<scalar>::mm,
        100000. * detray::unit<scalar>::mm, mat, thickness);

    /***************************
     * Generate simulation data
     ***************************/

    constexpr unsigned int theta_steps{10};
    constexpr unsigned int phi_steps{10};
    const vector3 x0{0, 0, 0};

    auto generator =
        detray::uniform_track_generator<traccc::free_track_parameters>(
            theta_steps, phi_steps, x0, p0, {M_PI / 2., M_PI / 2.},
            {phi0, phi0});

    // Smearing value for measurements
    detray::measurement_smearer<scalar> meas_smearer(
        50 * detray::unit<scalar>::um, 50 * detray::unit<scalar>::um);

    // Run simulator
    std::string file_path =
        std::to_string(p0) + "_GeV_" + std::to_string(phi0) + "_phi/";

    const std::string full_path =
        io::data_directory() + output_directory + "/" + file_path;

    boost::filesystem::create_directories(full_path);

    auto sim =
        detray::simulator(events, det, generator, meas_smearer, full_path);
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
    desc.add_options()("p0", po::value<scalar>()->required(),
                       "initial momentum in GeV/c");
    desc.add_options()("phi0", po::value<scalar>()->required(), "initial phi");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    auto output_directory = vm["output_directory"].as<std::string>();
    auto events = vm["events"].as<unsigned int>();
    auto p0 = vm["p0"].as<scalar>();
    auto phi0 = vm["phi0"].as<scalar>();

    std::cout << "Running " << argv[0] << " " << output_directory << " "
              << events << "  " << p0 << "  " << phi0 << std::endl;

    return simulate(output_directory, events, p0, phi0);
}
