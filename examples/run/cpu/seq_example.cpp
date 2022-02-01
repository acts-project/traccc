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

// System include(s).
#include <iostream>

int seq_run(const std::string& detector_file, const std::string& cells_dir,
            unsigned int events) {

    // Read the surface transforms
    auto surface_transforms = traccc::read_geometry(detector_file);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_internal_spacepoints = 0;
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
        n_seeds += seeds.total_size();

        /*------------
             Writer
          ------------*/

        traccc::write_measurements(event, measurements_per_event);
        traccc::write_spacepoints(event, spacepoints_per_event);
        traccc::write_seeds(event, spacepoints_per_event, seeds);
        traccc::write_estimated_track_parameters(event, params);
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells from " << n_modules
              << " modules" << std::endl;
    std::cout << "- created " << n_measurements << " measurements. "
              << std::endl;
    std::cout << "- created " << n_spacepoints << " space points. "
              << std::endl;
    std::cout << "- created " << n_internal_spacepoints
              << " internal spacepoints" << std::endl;
    std::cout << "- created " << n_seeds << " seeds" << std::endl;
    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << "./seq_example <detector_file> <cell_directory> <events>"
                  << std::endl;
        return -1;
    }

    auto detector_file = std::string(argv[1]);
    auto cell_directory = std::string(argv[2]);
    auto events = std::atoi(argv[3]);

    std::cout << "Running ./seq_exammple " << detector_file << " "
              << cell_directory << " " << events << std::endl;
    return seq_run(detector_file, cell_directory, events);
}
