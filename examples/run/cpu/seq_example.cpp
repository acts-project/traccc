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

// options
#include "traccc/seq_input_options.hpp"
#include "traccc/throw_exception.hpp"

// System include(s).
#include <exception>
#include <iostream>

int seq_run(const traccc::seq_input_config& i_cfg) {

    // Read the surface transforms
    auto surface_transforms = traccc::read_geometry(i_cfg.detector_file);

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

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});
    sd_performance_writer.add_cache("CPU");

    // Loop over events
    for (unsigned int event = 0; event < i_cfg.events; ++event) {

        // Read the cells from the relevant event file
        traccc::host_cell_container cells_per_event =
            traccc::read_cells_from_event(event + i_cfg.skip,
                                          i_cfg.cell_directory,
                                          surface_transforms, host_mr);

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

        if (i_cfg.check_seeding_performance) {
            traccc::event_map evt_map(event, i_cfg.detector_file,
                                      i_cfg.cell_directory, i_cfg.hit_directory,
                                      i_cfg.particle_directory, host_mr);

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
    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    traccc::add_seq_input_options(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Read options
    auto seq_input_cfg = traccc::read_seq_input_options(vm);

    // Check exception
    auto exception = traccc::throw_exception(desc, vm);
    if (exception != traccc::no_exception) {
        return exception;
    }

    std::cout << "Running " << argv[0] << " " << seq_input_cfg.detector_file
              << " " << seq_input_cfg.hit_directory << " "
              << seq_input_cfg.events << std::endl;

    return seq_run(seq_input_cfg);
}
