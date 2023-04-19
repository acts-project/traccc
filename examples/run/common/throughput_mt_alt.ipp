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
#include "traccc/io/demonstrator_alt_edm.hpp"
#include "traccc/io/read.hpp"

// Performance measurement include(s).
#include "traccc/performance/throughput.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/performance/timing_info.hpp"

// Baseline chain include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

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
int throughput_mt_alt(std::string_view description, int argc, char* argv[],
                      bool use_host_caching) {

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
    HOST_MR uncached_host_mr;

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

    // Set up cached memory resources on top of the host memory resource
    // separately for each CPU thread.
    std::vector<std::unique_ptr<vecmem::binary_page_memory_resource>>
        cached_host_mrs{mt_cfg.threads + 1};

    // Set up the full-chain algorithm(s). One for each thread.
    std::vector<FULL_CHAIN_ALG> algs;
    algs.reserve(mt_cfg.threads + 1);
    for (std::size_t i = 0; i < mt_cfg.threads + 1; ++i) {

        cached_host_mrs.at(i) =
            std::make_unique<vecmem::binary_page_memory_resource>(
                uncached_host_mr);
        vecmem::memory_resource& alg_host_mr =
            use_host_caching
                ? static_cast<vecmem::memory_resource&>(
                      *(cached_host_mrs.at(i)))
                : static_cast<vecmem::memory_resource&>(uncached_host_mr);
        algs.push_back(
            {alg_host_mr, throughput_cfg.target_cells_per_partition});
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
        for (std::size_t i = 0; i < throughput_cfg.cold_run_events; ++i) {

            // Choose which event to process.
            const std::size_t event =
                std::rand() % throughput_cfg.loaded_events;

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

    constexpr std::size_t intervalStoredResults = 1000;
    std::vector<std::pair<std::size_t, typename FULL_CHAIN_ALG::output_type>>
        storedResults;
    storedResults.reserve(
        (throughput_cfg.processed_events + intervalStoredResults - 1) /
        intervalStoredResults);

    {
        // Measure the total time of execution.
        performance::timer t{"Event processing", times};

        // Process the requested number of events.
        for (std::size_t i = 0; i < throughput_cfg.processed_events; ++i) {

            // Choose which event to process.
            const std::size_t event =
                std::rand() % throughput_cfg.loaded_events;

            // Launch the processing of the event.
            arena.execute([&, event, i]() {
                group.run([&, event, i]() {
                    auto track_params =
                        algs.at(tbb::this_task_arena::current_thread_index())(
                            input[event].cells, input[event].modules);
                    rec_track_params.fetch_add(track_params.size());
                    if (i % intervalStoredResults == 0) {
                        storedResults.push_back(
                            {event, std::move(track_params)});
                    }
                });
            });
        }

        // Wait for all tasks to finish.
        group.wait();
    }

    std::cout << "Finished event processing." << std::endl;

    // Delete the algorithms and host memory caches explicitly before their
    // parent object would go out of scope.
    algs.clear();
    cached_host_mrs.clear();

    // Memory resource used by the EDM.
    traccc::clusterization_algorithm ca(uncached_host_mr);
    traccc::spacepoint_formation sf(uncached_host_mr);
    traccc::seeding_algorithm sa(uncached_host_mr);
    traccc::track_params_estimation tp(uncached_host_mr);

    for (std::size_t i = 0; i < storedResults.size(); ++i) {
        auto sp = sf(ca(input[storedResults[i].first].cells,
                        input[storedResults[i].first].modules),
                     input[storedResults[i].first].modules);
        auto trackParams = tp(sp, sa(sp));
        // Compare the track parameters made on the host and on the device.
        traccc::collection_comparator<traccc::bound_track_parameters>
            compare_track_parameters{"track parameters", {},        "baseline",
                                     "throughput run",   std::cout, {0.0001}};
        compare_track_parameters(vecmem::get_data(trackParams),
                                 vecmem::get_data(storedResults[i].second));
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

    // Print results to log file
    if (throughput_cfg.log_file != "\0") {
        std::ofstream logFile;
        logFile.open(throughput_cfg.log_file, std::fstream::app);
        logFile << "\"" << throughput_cfg.input_directory << "\""
                << "," << mt_cfg.threads << "," << throughput_cfg.loaded_events
                << "," << throughput_cfg.cold_run_events << ","
                << throughput_cfg.processed_events << ","
                << throughput_cfg.target_cells_per_partition << ","
                << times.get_time("Warm-up processing").count() << ","
                << times.get_time("Event processing").count() << std::endl;
        logFile.close();
    }

    // Return gracefully.
    return 0;
}

}  // namespace traccc
