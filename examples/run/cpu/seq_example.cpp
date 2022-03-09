/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// io
#include "traccc/io/csv.hpp"
#include "traccc/io/reader.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/writer.hpp"

// algorithms
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"
#include "traccc/track_finding/seeding_algorithm.hpp"

// performance
#include "traccc/efficiency/seeding_performance_writer.hpp"

// Boost
#include <boost/program_options.hpp>

// System include(s).
#include <exception>
#include <iostream>

namespace po = boost::program_options;

int seq_run(const std::string& detector_file, const std::string& cells_dir,
            unsigned int events, const std::string& hit_dir,
            const std::string& particle_dir, const bool check_performance) {

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});
    sd_performance_writer.add_cache("CPU");

    // Read the surface transforms
    auto surface_transforms = traccc::read_geometry(detector_file);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    traccc::clusterization_algorithm ca(host_mr);
    traccc::seeding_algorithm sa(host_mr);
    traccc::track_params_estimation tp(host_mr);

    // Loop over events
    for (unsigned int event = 0; event < events; ++event) {

        // Read the cells from the relevant event file
        traccc::host_cell_container cells_per_event =
            traccc::read_cells_from_event(event, cells_dir, surface_transforms,
                                          host_mr);

        /*-------------------
            Clusterization
          -------------------*/

        auto ca_result = ca(cells_per_event);
        auto& measurements_per_event = ca_result.first;
        auto& spacepoints_per_event = ca_result.second;

        /*-----------------------
          Seeding algorithm
          -----------------------*/

        auto seeds = sa(spacepoints_per_event);

        /*----------------------------
          Track params estimation
          ----------------------------*/

        auto tp_output = tp(spacepoints_per_event, seeds);
        auto& params = tp_output;

        /*----------------------------
          Statistics
          ----------------------------*/

        n_modules += cells_per_event.size();
        n_cells += cells_per_event.total_size();
        n_measurements += measurements_per_event.total_size();
        n_spacepoints += spacepoints_per_event.total_size();
        n_seeds += seeds.size();

        /*------------
             Writer
          ------------*/

        if (check_performance) {
            traccc::event_map evt_map(event, detector_file, cells_dir, hit_dir,
                                      particle_dir, host_mr);

            sd_performance_writer.write("CPU", seeds, spacepoints_per_event,
                                        evt_map);
        }

        traccc::write_measurements(event, measurements_per_event);
        traccc::write_spacepoints(event, spacepoints_per_event);
        traccc::write_seeds(event, spacepoints_per_event, seeds);
        traccc::write_estimated_track_parameters(event, params);
    }

    sd_performance_writer.finalize();

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells from " << n_modules
              << " modules" << std::endl;
    std::cout << "- created " << n_measurements << " measurements. "
              << std::endl;
    std::cout << "- created " << n_spacepoints << " space points. "
              << std::endl;
    std::cout << "- created " << n_seeds << " seeds" << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {

    // Set up the program options.
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Give some help with the program's options");
    desc.add_options()("detector_file", po::value<std::string>()->required(),
                       "specify detector file");
    desc.add_options()("cell_directory", po::value<std::string>()->required(),
                       "specify the directory of cell files");
    desc.add_options()("events", po::value<int>()->required(),
                       "number of events");
    desc.add_options()("hit_directory",
                       po::value<std::string>()->default_value(""),
                       "specify the directory of hit files");
    desc.add_options()("particle_directory",
                       po::value<std::string>()->default_value(""),
                       "specify the directory of particle files");

    // Interpret the program options.
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Print a help message if the user asked for it.
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    // Handle any and all errors.
    try {
        po::notify(vm);
    } catch (const std::exception& ex) {
        std::cerr << "Couldn't interpret command line options because of:\n\n"
                  << ex.what() << "\n\n"
                  << desc << std::endl;
        return 1;
    }

    auto detector_file = vm["detector_file"].as<std::string>();
    auto cell_directory = vm["cell_directory"].as<std::string>();
    auto events = vm["events"].as<int>();
    auto hit_directory = vm["hit_directory"].as<std::string>();
    auto particle_directory = vm["particle_directory"].as<std::string>();
    auto check_performance =
        vm.count("hit_directory") && vm.count("particle_directory");

    std::cout << "Running " << argv[0] << " " << detector_file << " "
              << cell_directory << " " << events << std::endl;

    return seq_run(detector_file, cell_directory, events, hit_directory,
                   particle_directory, check_performance);
}
