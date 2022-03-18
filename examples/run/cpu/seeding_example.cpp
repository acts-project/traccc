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

// options
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/seeding_input_options.hpp"

// System include(s).
#include <iostream>

int seq_run(const traccc::seeding_input_config& i_cfg) {

    // Read the surface transforms
    auto surface_transforms = traccc::read_geometry(i_cfg.detector_file);

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
    for (unsigned int event = i_cfg.skip; event < i_cfg.events + i_cfg.skip;
         ++event) {

        // Read the hits from the relevant event file
        traccc::host_spacepoint_container spacepoints_per_event =
            traccc::read_spacepoints_from_event(event, i_cfg.hit_directory,
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

        if (i_cfg.check_seeding_performance) {
            traccc::event_map evt_map(event, i_cfg.detector_file,
                                      i_cfg.hit_directory,
                                      i_cfg.particle_directory, host_mr);
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
    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    traccc::seeding_input_config seeding_input_cfg(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Read options
    seeding_input_cfg.read(vm);

    // Check exception
    auto exception = traccc::handle_argument_errors(desc, vm);
    if (exception != traccc::no_exception) {
        return exception;
    }

    std::cout << "Running " << argv[0] << " " << seeding_input_cfg.detector_file
              << " " << seeding_input_cfg.hit_directory << " "
              << seeding_input_cfg.events << std::endl;

    return seq_run(seeding_input_cfg);
}
