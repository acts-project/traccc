/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Command line option include(s).
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/threading.hpp"
#include "traccc/options/throughput.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"

// I/O include(s).
#include "traccc/io/demonstrator_edm.hpp"
#include "traccc/io/read.hpp"

// Performance measurement include(s).
#include "traccc/performance/throughput.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/performance/timing_info.hpp"

// VecMem include(s).
#include <vecmem/memory/binary_page_memory_resource.hpp>

// TBB include(s).
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

// System include(s).
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace traccc {

template <typename FULL_CHAIN_ALG, typename HOST_MR>
int throughput_mt(std::string_view description, int argc, char* argv[],
                  bool use_host_caching) {

    // Convenience typedef.
    namespace po = boost::program_options;

    // Read in the command line options.
    po::options_description desc{description.data()};
    desc.add_options()("help,h", "Give help with the program's options");
    opts::detector detector_opts{desc};
    opts::input_data input_opts{desc};
    opts::clusterization clusterization_opts{desc};
    opts::throughput throughput_opts{desc};
    opts::threading threading_opts{desc};

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    handle_argument_errors(vm, desc);

    detector_opts.read(vm);
    input_opts.read(vm);
    clusterization_opts.read(vm);
    throughput_opts.read(vm);
    threading_opts.read(vm);

    // Greet the user.
    std::cout << "\n"
              << description << "\n\n"
              << detector_opts << "\n"
              << input_opts << "\n"
              << clusterization_opts << "\n"
              << throughput_opts << "\n"
              << threading_opts << "\n"
              << std::endl;

    // Set seeding config
    // @FIXME: Seeding config should be configured by options
    seedfinder_config finder_config;
    spacepoint_grid_config grid_config(finder_config);
    seedfilter_config filter_config;

    // Set up the timing info holder.
    performance::timing_info times;

    // Set up the TBB arena and thread group.
    tbb::global_control global_thread_limit(
        tbb::global_control::max_allowed_parallelism,
        threading_opts.threads + 1);
    tbb::task_arena arena{static_cast<int>(threading_opts.threads), 0};
    tbb::task_group group;

    // Memory resource to use in the test.
    HOST_MR uncached_host_mr;

    // Read in all input events into memory.
    demonstrator_input input(&uncached_host_mr);

    {
        performance::timer t{"File reading", times};
        // Create empty inputs using the correct memory resource
        for (std::size_t i = 0; i < input_opts.events; ++i) {
            input.push_back(demonstrator_input::value_type(&uncached_host_mr));
        }
        // Read event data into input vector
        io::read(input, input_opts.events, input_opts.directory,
                 detector_opts.detector_file, detector_opts.digitization_file,
                 input_opts.format);
    }

    // Set up cached memory resources on top of the host memory resource
    // separately for each CPU thread.
    std::vector<std::unique_ptr<vecmem::binary_page_memory_resource> >
        cached_host_mrs{threading_opts.threads + 1};

    // Set up the full-chain algorithm(s). One for each thread.
    std::vector<FULL_CHAIN_ALG> algs;
    algs.reserve(threading_opts.threads + 1);
    for (std::size_t i = 0; i < threading_opts.threads + 1; ++i) {

        cached_host_mrs.at(i) =
            std::make_unique<vecmem::binary_page_memory_resource>(
                uncached_host_mr);
        vecmem::memory_resource& alg_host_mr =
            use_host_caching
                ? static_cast<vecmem::memory_resource&>(
                      *(cached_host_mrs.at(i)))
                : static_cast<vecmem::memory_resource&>(uncached_host_mr);
        algs.push_back({alg_host_mr,
                        clusterization_opts.target_cells_per_partition,
                        finder_config, grid_config, filter_config});
    }

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
        for (std::size_t i = 0; i < throughput_opts.cold_run_events; ++i) {

            // Choose which event to process.
            const std::size_t event = std::rand() % input_opts.events;

            // Launch the processing of the event.
            arena.execute([&, event]() {
                group.run([&, event]() {
                    rec_track_params.fetch_add(
                        algs.at(tbb::this_task_arena::current_thread_index())(
                                input[event].cells, input[event].modules)
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
        for (std::size_t i = 0; i < throughput_opts.processed_events; ++i) {

            // Choose which event to process.
            const std::size_t event = std::rand() % input_opts.events;

            // Launch the processing of the event.
            arena.execute([&, event]() {
                group.run([&, event]() {
                    rec_track_params.fetch_add(
                        algs.at(tbb::this_task_arena::current_thread_index())(
                                input[event].cells, input[event].modules)
                            .size());
                });
            });
        }

        // Wait for all tasks to finish.
        group.wait();
    }

    // Delete the algorithms and host memory caches explicitly before their
    // parent object would go out of scope.
    algs.clear();
    cached_host_mrs.clear();

    // Print some results.
    std::cout << "Reconstructed track parameters: " << rec_track_params.load()
              << std::endl;
    std::cout << "Time totals:" << std::endl;
    std::cout << times << std::endl;
    std::cout << "Throughput:" << std::endl;
    std::cout << performance::throughput{throughput_opts.cold_run_events, times,
                                         "Warm-up processing"}
              << "\n"
              << performance::throughput{throughput_opts.processed_events,
                                         times, "Event processing"}
              << std::endl;

    // Print results to log file
    if (throughput_opts.log_file != "\0") {
        std::ofstream logFile;
        logFile.open(throughput_opts.log_file, std::fstream::app);
        logFile << "\"" << input_opts.directory << "\""
                << "," << threading_opts.threads << "," << input_opts.events
                << "," << throughput_opts.cold_run_events << ","
                << throughput_opts.processed_events << ","
                << clusterization_opts.target_cells_per_partition << ","
                << times.get_time("Warm-up processing").count() << ","
                << times.get_time("Event processing").count() << std::endl;
        logFile.close();
    }

    // Return gracefully.
    return 0;
}

}  // namespace traccc
