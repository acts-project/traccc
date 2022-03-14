/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// io
#include "traccc/io/csv.hpp"
#include "traccc/io/reader.hpp"
#include "traccc/io/writer.hpp"

// algorithms
#include "traccc/seeding/track_params_estimation.hpp"
#include "traccc/track_finding/seeding_algorithm.hpp"

// performance
#include "traccc/efficiency/seeding_performance_writer.hpp"

// Boost
#include <boost/program_options.hpp>

// System include(s).
#include <iostream>

namespace po = boost::program_options;

int seq_run(const std::string& detector_file, const std::string& hit_dir,
            unsigned int events, const std::string& particle_dir,
            const bool check_performance) {

    // Read the surface transforms
    auto surface_transforms = traccc::read_geometry(detector_file);

    // Output stats
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    traccc::seeding_algorithm sa(host_mr);
    traccc::track_params_estimation tp(host_mr);

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});
    sd_performance_writer.add_cache("CPU");

    // Loop over events
    for (unsigned int event = 0; event < events; ++event) {

        // Read the hits from the relevant event file
        traccc::host_spacepoint_container spacepoints_per_event =
            traccc::read_spacepoints_from_event(event, hit_dir,
                                                surface_transforms, host_mr);

        /*----------------
             Seeding
          ---------------*/

        auto seeds = sa(spacepoints_per_event);

        /*----------------
             Statistics
          ---------------*/

        n_spacepoints += spacepoints_per_event.total_size();
        n_seeds += seeds.size();

        /*------------
          Writer
          ------------*/

        if (check_performance) {
            traccc::event_map evt_map(event, detector_file, hit_dir,
                                      particle_dir, host_mr);
            sd_performance_writer.write("CPU", seeds, spacepoints_per_event,
                                        evt_map);
        }
    }

    sd_performance_writer.finalize();

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints" << std::endl;
    std::cout << "- created (cpu)  " << n_seeds << " seeds" << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()("detector_file", po::value<std::string>()->required(),
                       "specify detector file");
    desc.add_options()("hit_directory", po::value<std::string>()->required(),
                       "specify the directory of hit files");
    desc.add_options()("events", po::value<int>()->required(),
                       "number of events");
    desc.add_options()("particle_directory",
                       po::value<std::string>()->default_value(""),
                       "specify the directory of particle files");

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
    auto hit_directory = vm["hit_directory"].as<std::string>();
    auto events = vm["events"].as<int>();
    auto particle_directory = vm["particle_directory"].as<std::string>();
    auto check_performance = (particle_directory != "");

    std::cout << "Running " << argv[0] << " " << detector_file << " "
              << hit_directory << " " << events << std::endl;

    return seq_run(detector_file, hit_directory, events, particle_directory,
                   check_performance);
}
