/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/geometry/detector.hpp"

// Command line option include(s).
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/threading.hpp"
#include "traccc/options/throughput.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"

// I/O include(s).
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/utils.hpp"

// Performance measurement include(s).
#include "traccc/performance/throughput.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/performance/timing_info.hpp"

// VecMem include(s).
#include <vecmem/memory/binary_page_memory_resource.hpp>

// TBB include(s).
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#include <tbb/global_control.h>
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <tbb/global_control.h>
#pragma GCC diagnostic pop
#else
#include <tbb/global_control.h>
#endif
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

// Indicators include(s).
#include <indicators/progress_bar.hpp>

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
    std::unique_ptr<const traccc::Logger> ilogger = traccc::getDefaultLogger(
        "ThroughputExample", traccc::Logging::Level::INFO);
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    // Program options.
    opts::detector detector_opts;
    opts::input_data input_opts;
    opts::clusterization clusterization_opts;
    opts::track_seeding seeding_opts;
    opts::track_finding finding_opts;
    opts::track_propagation propagation_opts;
    opts::track_fitting fitting_opts;
    opts::throughput throughput_opts;
    opts::threading threading_opts;
    opts::program_options program_opts{
        description,
        {detector_opts, input_opts, clusterization_opts, seeding_opts,
         finding_opts, propagation_opts, fitting_opts, throughput_opts,
         threading_opts},
        argc,
        argv,
        logger().cloneWithSuffix("Options")};

    // Set up the timing info holder.
    performance::timing_info times;

    // Memory resource to use in the test.
    HOST_MR uncached_host_mr;

    // Construct the detector description object.
    traccc::silicon_detector_description::host det_descr{uncached_host_mr};
    traccc::io::read_detector_description(
        det_descr, detector_opts.detector_file, detector_opts.digitization_file,
        (detector_opts.use_detray_detector ? traccc::data_format::json
                                           : traccc::data_format::csv));

    // Construct a Detray detector object, if supported by the configuration.
    traccc::default_detector::host detector{uncached_host_mr};
    if (detector_opts.use_detray_detector) {
        traccc::io::read_detector(
            detector, uncached_host_mr, detector_opts.detector_file,
            detector_opts.material_file, detector_opts.grid_file);
    }

    // Read in all input events into memory.
    vecmem::vector<edm::silicon_cell_collection::host> input{&uncached_host_mr};
    {
        performance::timer t{"File reading", times};
        // Set up the container for the input events.
        input.reserve(input_opts.events);
        const std::size_t first_event = input_opts.skip;
        const std::size_t last_event = input_opts.skip + input_opts.events;
        for (std::size_t i = first_event; i < last_event; ++i) {
            input.push_back({uncached_host_mr});
        }
        // Read the input cells into memory in parallel.
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>{first_event, last_event},
            [&](const tbb::blocked_range<std::size_t>& event_range) {
                for (std::size_t event = event_range.begin();
                     event != event_range.end(); ++event) {
                    static constexpr bool DEDUPLICATE = true;
                    io::read_cells(input.at(event - input_opts.skip), event,
                                   input_opts.directory, logger().clone(),
                                   &det_descr, input_opts.format, DEDUPLICATE,
                                   input_opts.use_acts_geom_source);
                }
            });
    }

    // Set up cached memory resources on top of the host memory resource
    // separately for each CPU thread.
    std::vector<std::unique_ptr<vecmem::binary_page_memory_resource> >
        cached_host_mrs;
    if (use_host_caching) {
        cached_host_mrs.reserve(threading_opts.threads + 1);
        for (std::size_t i = 0; i < threading_opts.threads + 1; ++i) {
            cached_host_mrs.push_back(
                std::make_unique<vecmem::binary_page_memory_resource>(
                    uncached_host_mr));
        }
    }

    typename FULL_CHAIN_ALG::clustering_algorithm::config_type clustering_cfg(
        clusterization_opts);

    // Algorithm configuration(s).
    detray::propagation::config propagation_config(propagation_opts);
    typename FULL_CHAIN_ALG::finding_algorithm::config_type finding_cfg(
        finding_opts);
    finding_cfg.propagation = propagation_config;

    typename FULL_CHAIN_ALG::fitting_algorithm::config_type fitting_cfg(
        fitting_opts);
    fitting_cfg.propagation = propagation_config;

    // Set up the full-chain algorithm(s). One for each thread.
    std::vector<FULL_CHAIN_ALG> algs;
    algs.reserve(threading_opts.threads + 1);
    for (std::size_t i = 0; i < threading_opts.threads + 1; ++i) {

        vecmem::memory_resource& alg_host_mr =
            use_host_caching
                ? static_cast<vecmem::memory_resource&>(
                      *(cached_host_mrs.at(i)))
                : static_cast<vecmem::memory_resource&>(uncached_host_mr);
        algs.push_back(
            {alg_host_mr,
             clustering_cfg,
             seeding_opts.seedfinder,
             {seeding_opts.seedfinder},
             seeding_opts.seedfilter,
             finding_cfg,
             fitting_cfg,
             det_descr,
             (detector_opts.use_detray_detector ? &detector : nullptr),
             logger().clone()});
    }

    // Set up the TBB arena and thread group. From here on out TBB is only
    // allowed to use the specified number of threads.
    tbb::global_control global_thread_limit(
        tbb::global_control::max_allowed_parallelism,
        threading_opts.threads + 1);
    tbb::task_arena arena{static_cast<int>(threading_opts.threads), 0};
    tbb::task_group group;

    // Seed the random number generator.
    if (throughput_opts.random_seed == 0u) {
        std::srand(static_cast<unsigned int>(std::time(0)));
    } else {
        std::srand(throughput_opts.random_seed);
    }

    // Dummy count uses output of tp algorithm to ensure the compiler
    // optimisations don't skip any step
    std::atomic_size_t rec_track_params = 0;

    // Cold Run events. To discard any "initialisation issues" in the
    // measurements.
    {
        // Set up a progress bar for the warm-up processing.
        indicators::ProgressBar progress_bar{
            indicators::option::BarWidth{50},
            indicators::option::PrefixText{"Warm-up processing "},
            indicators::option::ShowPercentage{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::MaxProgress{throughput_opts.cold_run_events}};

        // Measure the time of execution.
        performance::timer t{"Warm-up processing", times};

        // Process the requested number of events.
        for (std::size_t i = 0; i < throughput_opts.cold_run_events; ++i) {

            // Choose which event to process.
            const std::size_t event =
                (throughput_opts.deterministic_event_order
                     ? i
                     : static_cast<std::size_t>(std::rand())) %
                input_opts.events;

            // Launch the processing of the event.
            arena.execute([&, event]() {
                group.run([&, event]() {
                    rec_track_params.fetch_add(algs.at(static_cast<std::size_t>(
                        tbb::this_task_arena::current_thread_index()))(
                                                       input[event])
                                                   .size());
                    progress_bar.tick();
                });
            });
        }

        // Wait for all tasks to finish.
        group.wait();
    }

    // Reset the dummy counter.
    rec_track_params = 0;

    {
        // Set up a progress bar for the event processing.
        indicators::ProgressBar progress_bar{
            indicators::option::BarWidth{50},
            indicators::option::PrefixText{"Event processing   "},
            indicators::option::ShowPercentage{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::MaxProgress{throughput_opts.processed_events}};

        // Measure the total time of execution.
        performance::timer t{"Event processing", times};

        // Process the requested number of events.
        for (std::size_t i = 0; i < throughput_opts.processed_events; ++i) {

            // Choose which event to process.
            const std::size_t event =
                (throughput_opts.deterministic_event_order
                     ? i
                     : static_cast<std::size_t>(std::rand())) %
                input_opts.events;

            // Launch the processing of the event.
            arena.execute([&, event]() {
                group.run([&, event]() {
                    rec_track_params.fetch_add(algs.at(static_cast<std::size_t>(
                        tbb::this_task_arena::current_thread_index()))(
                                                       input[event])
                                                   .size());
                    progress_bar.tick();
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
    TRACCC_INFO("Reconstructed track parameters: " << rec_track_params.load());
    TRACCC_INFO("Time totals: " << times);

    performance::throughput throughput_wu{throughput_opts.cold_run_events,
                                          times, "Warm-up processing"};
    performance::throughput throughput_pr{throughput_opts.processed_events,
                                          times, "Event processing"};

    TRACCC_INFO("Throughput:" << throughput_wu << "\n" << throughput_pr);

    // Print results to log file
    if (throughput_opts.log_file != "\0") {
        std::ofstream logFile;
        logFile.open(throughput_opts.log_file, std::fstream::app);
        logFile << "\"" << input_opts.directory << "\""
                << "," << threading_opts.threads << "," << input_opts.events
                << "," << throughput_opts.cold_run_events << ","
                << throughput_opts.processed_events << ","
                << times.get_time("Warm-up processing").count() << ","
                << times.get_time("Event processing").count() << std::endl;
        logFile.close();
    }

    // Return gracefully.
    return 0;
}

}  // namespace traccc
