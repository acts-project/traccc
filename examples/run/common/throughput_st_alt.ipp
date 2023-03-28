/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Command line option include(s).
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/throughput_options.hpp"

// I/O include(s).
#include "traccc/io/demonstrator_alt_edm.hpp"
#include "traccc/io/read.hpp"

// Performance measurement include(s).
#include "traccc/performance/throughput.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/performance/timing_info.hpp"

// VecMem include(s).
#include <vecmem/memory/binary_page_memory_resource.hpp>

// System include(s).
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>

namespace traccc {

template <typename FULL_CHAIN_ALG, typename HOST_MR>
int throughput_st_alt(std::string_view description, int argc, char* argv[],
                      bool use_host_caching) {

    // Convenience typedef.
    namespace po = boost::program_options;

    // Read in the command line options.
    po::options_description desc{description.data()};
    desc.add_options()("help,h", "Give help with the program's options");
    throughput_options throughput_cfg{desc};

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    handle_argument_errors(vm, desc);

    throughput_cfg.read(vm);

    // Greet the user.
    std::cout << "\n"
              << description << "\n\n"
              << throughput_cfg << "\n"
              << std::endl;

    // Set up the timing info holder.
    performance::timing_info times;

    // Memory resource to use in the test.
    HOST_MR uncached_host_mr;
    std::unique_ptr<vecmem::binary_page_memory_resource> cached_host_mr =
        std::make_unique<vecmem::binary_page_memory_resource>(uncached_host_mr);

    vecmem::memory_resource& alg_host_mr =
        use_host_caching
            ? static_cast<vecmem::memory_resource&>(*cached_host_mr)
            : static_cast<vecmem::memory_resource&>(uncached_host_mr);

    // Read in all input events into memory.
    alt_demonstrator_input input;
    {
        performance::timer t{"File reading", times};
        for (unsigned int event = 0; event < throughput_cfg.loaded_events;
             ++event) {
            input = io::read(
                throughput_cfg.loaded_events, throughput_cfg.input_directory,
                throughput_cfg.detector_file,
                throughput_cfg.digitization_config_file,
                throughput_cfg.input_data_format, &uncached_host_mr);
        }
    }

    // Set up the full-chain algorithm.
    std::unique_ptr<FULL_CHAIN_ALG> alg = std::make_unique<FULL_CHAIN_ALG>(
        alg_host_mr, throughput_cfg.target_cells_per_partition);

    // Seed the random number generator.
    std::srand(std::time(0));

    // Dummy count uses output of tp algorithm to ensure the compiler
    // optimisations don't skip any step
    std::size_t rec_track_params = 0;

    // Cold Run events. To discard any "initialisation issues" in the
    // measurements.
    {
        // Measure the time of execution.
        performance::timer t{"Warm-up processing", times};

        // Process the requested number of events.
        for (std::size_t i = 0; i < throughput_cfg.cold_run_events; ++i) {

            // Choose which event to process.
            const std::size_t event =
                std::rand() % throughput_cfg.loaded_events;

            // Process one event.
            rec_track_params +=
                (*alg)(input[event].cells, input[event].modules).size();
        }
    }

    // Reset the dummy counter.
    rec_track_params = 0;

    {
        // Measure the total time of execution.
        performance::timer t{"Event processing", times};

        // Process the requested number of events.
        for (std::size_t i = 0; i < throughput_cfg.processed_events; ++i) {

            // Choose which event to process.
            const std::size_t event =
                std::rand() % throughput_cfg.loaded_events;

            // Process one event.
            rec_track_params +=
                (*alg)(input[event].cells, input[event].modules).size();
        }
    }

    // Explicitly delete the objects in the correct order.
    alg.reset();
    cached_host_mr.reset();

    // Print some results.
    std::cout << "Reconstructed track parameters: " << rec_track_params
              << std::endl;
    std::cout << "Time totals:" << std::endl;
    std::cout << times << std::endl;
    std::cout << "Throughput:" << std::endl;
    std::cout << performance::throughput{throughput_cfg.cold_run_events, times,
                                         "Warm-up processing"}
              << "\n"
              << performance::throughput{throughput_cfg.processed_events, times,
                                         "Event processing"}
              << std::endl;

    // Return gracefully.
    return 0;
}

}  // namespace traccc
