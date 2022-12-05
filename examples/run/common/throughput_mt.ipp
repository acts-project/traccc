/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Command line option include(s).
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/mt_options.hpp"
#include "traccc/options/throughput_options.hpp"

// I/O include(s).
#include "traccc/io/read.hpp"

// Performance measurement include(s).
#include "traccc/performance/throughput.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/performance/timing_info.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// TBB include(s).
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

// System include(s).
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

namespace traccc {

template <typename FULL_CHAIN_ALG>
int throughput_mt(std::string_view description, int argc, char* argv[]) {

    // Convenience typedef.
    namespace po = boost::program_options;

    // Read in the command line options.
    po::options_description desc{description.data()};
    desc.add_options()("help,h", "Give help with the program's options");
    throughput_options throughput_cfg{desc};
    mt_options mt_cfg{desc};

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    handle_argument_errors(vm, desc);

    throughput_cfg.read(vm);
    mt_cfg.read(vm);

    // Greet the user.
    std::cout << "\n"
              << description << "\n\n"
              << throughput_cfg << "\n"
              << mt_cfg << "\n"
              << std::endl;

    // Set up the timing info holder.
    performance::timing_info times;

    // Set up the TBB arena and thread group.
    tbb::global_control global_thread_limit(
        tbb::global_control::max_allowed_parallelism, mt_cfg.threads + 1);
    tbb::task_arena arena{static_cast<int>(mt_cfg.threads), 0};
    tbb::task_group group;

    // Memory resource to use in the test.
    vecmem::host_memory_resource host_mr;

    // Read in all input events into memory.
    demonstrator_input cells;
    {
        performance::timer t{"File reading", times};
        cells = io::read(throughput_cfg.loaded_events,
                         throughput_cfg.input_directory,
                         throughput_cfg.detector_file,
                         throughput_cfg.digitization_config_file,
                         throughput_cfg.input_data_format, &host_mr);
    }

    // Set up the full-chain algorithm(s). One for each thread.
    std::vector<FULL_CHAIN_ALG> alg{mt_cfg.threads + 1, {host_mr}};

    // Seed the random number generator.
    std::srand(std::time(0));

    // Dummy count uses output of tp algorithm to ensure the compiler
    // optimisations don't skip any step
    std::atomic_size_t rec_track_params = 0;

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

            // Launch the processing of the event.
            arena.execute([&]() {
                group.run([&]() {
                    rec_track_params.fetch_add(
                        alg.at(tbb::this_task_arena::current_thread_index())(
                               cells[event])
                            .size());
                });
            });
        }

        // Wait for all tasks to finish.
        group.wait();
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

            // Launch the processing of the event.
            arena.execute([&]() {
                group.run([&]() {
                    rec_track_params.fetch_add(
                        alg.at(tbb::this_task_arena::current_thread_index())(
                               cells[event])
                            .size());
                });
            });
        }

        // Wait for all tasks to finish.
        group.wait();
    }

    // Print some results.
    std::cout << "Reconstructed track parameters: " << rec_track_params.load()
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
