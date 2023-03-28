/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/edm/alt_measurement.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/pixel_data.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Boost
#include <boost/program_options.hpp>

// OpenMP
#ifdef _OPENMP
#include "omp.h"
#endif

// System include(s).
#include <chrono>
#include <exception>
#include <iostream>

namespace po = boost::program_options;

int par_run(const std::string &detector_file,
            const std::string &digi_config_file, const std::string &cells_dir,
            unsigned int events) {

    // Read the surface transforms
    auto surface_transforms = traccc::io::read_geometry(detector_file);

    // Read the digitization configuration file
    auto digi_cfg = traccc::io::read_digitization_config(digi_config_file);

    // Memory resource used by the EDM.
    vecmem::host_memory_resource resource;

    // Algorithms
    traccc::clusterization_algorithm ca(resource);
    traccc::spacepoint_formation sf(resource);

    // Output stats
    uint64_t n_modules = 0;
    uint64_t n_cells = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;

#pragma omp parallel for reduction (+:n_modules, n_cells, n_measurements, n_spacepoints)
    // Loop over events
    for (unsigned int event = 0; event < events; ++event) {

        // Read the cells from the relevant event file
        auto readOut =
            traccc::io::read_cells(event, cells_dir, traccc::data_format::csv,
                                   &surface_transforms, &digi_cfg, &resource);
        traccc::cell_collection_types::host &cells_per_event = readOut.cells;
        traccc::cell_module_collection_types::host &modules_per_event =
            readOut.modules;

        /*-------------------
            Clusterization
          -------------------*/

        auto measurements_per_event = ca(cells_per_event, modules_per_event);

        /*------------------------
            Spacepoint formation
          ------------------------*/

        auto spacepoints_per_event =
            sf(measurements_per_event, modules_per_event);

        /*----------------------------
          Statistics
          ----------------------------*/

        n_modules += modules_per_event.size();
        n_cells += cells_per_event.size();
        n_measurements += measurements_per_event.size();
        n_spacepoints += spacepoints_per_event.size();
    }

#pragma omp critical

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells from " << n_modules
              << " modules" << std::endl;
    std::cout << "- created " << n_measurements << " measurements. "
              << std::endl;
    std::cout << "- created " << n_spacepoints << " spacepoints. " << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char *argv[]) {

    // Set up the program options.
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Give some help with the program's options");
    desc.add_options()("detector_file", po::value<std::string>()->required(),
                       "specify detector file");
    desc.add_options()("digitization_config_file",
                       po::value<std::string>()->required(),
                       "specify digitization configuration file");
    desc.add_options()("cell_directory", po::value<std::string>()->required(),
                       "specify the directory of cell files");
    desc.add_options()("events", po::value<int>()->required(),
                       "number of events");

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
    } catch (const std::exception &ex) {
        std::cerr << "Couldn't interpret command line options because of:\n\n"
                  << ex.what() << "\n\n"
                  << desc << std::endl;
        return 1;
    }

    auto detector_file = vm["detector_file"].as<std::string>();
    auto digi_config_file = vm["digitization_config_file"].as<std::string>();
    auto cell_directory = vm["cell_directory"].as<std::string>();
    auto events = vm["events"].as<int>();

    std::cout << "Running " << argv[0] << " " << detector_file << " "
              << cell_directory << " " << events << std::endl;

    auto start = std::chrono::system_clock::now();
    auto result =
        par_run(detector_file, digi_config_file, cell_directory, events);
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Execution time: " << diff.count() << " sec." << std::endl;
    return result;
}
