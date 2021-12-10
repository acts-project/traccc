/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <iostream>

// io
#include "io/csv.hpp"
#include "io/reader.hpp"
#include "io/utils.hpp"
#include "io/writer.hpp"

// algorithms
#include "clusterization/clusterization_algorithm.hpp"
#include "seeding/track_params_estimation.hpp"
#include "track_finding/seeding_algorithm.hpp"

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

    traccc::clusterization_algorithm ca;
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

        auto sa_output = sa(spacepoints_per_event);
        auto& internal_sp_per_event = sa_output.first;
        auto& seeds = sa_output.second;

        /*----------------------------
          Track params estimation
          ----------------------------*/

        auto tp_output = tp(seeds);
        auto& params = tp_output;

        /*----------------------------
          Statistics
          ----------------------------*/

        n_modules += cells_per_event.size();
        n_cells += cells_per_event.total_size();
        n_measurements += measurements_per_event.total_size();
        n_spacepoints += spacepoints_per_event.total_size();
        n_internal_spacepoints += internal_sp_per_event.total_size();
        n_seeds += seeds.total_size();

        /*------------
             Writer
          ------------*/

        traccc::write_measurements(event, measurements_per_event);
        traccc::write_spacepoints(event, spacepoints_per_event);
        traccc::write_internal_spacepoints(event, internal_sp_per_event);
        traccc::write_seeds(event, seeds);
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
