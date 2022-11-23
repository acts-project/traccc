/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// io
#include "traccc/io/read.hpp"

// algorithms
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// performance
#include "traccc/performance/throughput.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/performance/timing_info.hpp"

// options
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/mt_options.hpp"
#include "traccc/options/throughput_options.hpp"

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

int main(int argc, char* argv[]) {

    // Convenience typedef.
    namespace po = boost::program_options;

    // Read in the command line options.
    po::options_description desc{"Multi-threaded host-only throughput tests"};
    desc.add_options()("help,h", "Give help with the program's options");
    traccc::throughput_options throughput_cfg{desc};
    traccc::mt_options mt_cfg{desc};

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    traccc::handle_argument_errors(vm, desc);

    throughput_cfg.read(vm);
    mt_cfg.read(vm);

    // Greet the user.
    std::cout << "\nMulti-threaded throughput test\n\n"
              << throughput_cfg << "\n"
              << mt_cfg << "\n"
              << std::endl;

    // Set up the timing info holder.
    traccc::performance::timing_info times;

    // Set up the TBB arena and thread group.
    tbb::global_control global_thread_limit(
        tbb::global_control::max_allowed_parallelism, mt_cfg.threads + 1);
    tbb::task_arena arena{static_cast<int>(mt_cfg.threads), 0};
    tbb::task_group group;

    // Memory resource to use in the test.
    vecmem::host_memory_resource host_mr;

    // Read in all input events into memory.
    traccc::demonstrator_input cells;
    {
        traccc::performance::timer t{"File reading", times};
        cells = traccc::io::read(throughput_cfg.loaded_events,
                                 throughput_cfg.input_directory,
                                 throughput_cfg.detector_file,
                                 throughput_cfg.digitization_config_file,
                                 throughput_cfg.input_data_format, &host_mr);
    }

    // Set up all of the algorithms.
    traccc::clusterization_algorithm ca{host_mr};
    traccc::spacepoint_formation sf{host_mr};
    traccc::seeding_algorithm sa{host_mr};
    traccc::track_params_estimation tp{host_mr};

    // Seed the random number generator.
    std::srand(std::time(0));

    // Dummy count uses output of tp algorithm to ensure the compiler
    // optimisations don't skip any step
    std::atomic_size_t rec_track_params = 0;

    // Cold Run events
    // These are not accounted for the performance measurement
    for (std::size_t i = 0; i < throughput_cfg.cold_run_events; ++i) {
        // Choose which event to process.
        const std::size_t event = std::rand() % throughput_cfg.loaded_events;

        // Launch the processing of the event.
        arena.execute([&]() {
            group.run([&]() {
                const traccc::cell_container_types::host& input = cells[event];
                auto spacepoints = sf(ca(input));
                auto track_params = tp(spacepoints, sa(spacepoints));
                rec_track_params.fetch_add(track_params.size());
            });
        });
    }

    // Wait for all tasks to finish.
    group.wait();

    {
        // Measure the total time of execution.
        traccc::performance::timer t{"Event processing", times};

        // Process the requested number of events.
        for (std::size_t i = 0; i < throughput_cfg.processed_events; ++i) {

            // Choose which event to process.
            const std::size_t event =
                std::rand() % throughput_cfg.loaded_events;

            // Launch the processing of the event.
            arena.execute([&]() {
                group.run([&]() {
                    const traccc::cell_container_types::host& input =
                        cells[event];
                    auto spacepoints = sf(ca(input));
                    auto track_params = tp(spacepoints, sa(spacepoints));
                    rec_track_params.fetch_add(track_params.size());
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
    std::cout
        << traccc::performance::throughput{throughput_cfg.processed_events,
                                           times, "Event processing"}
        << std::endl;

    // Return gracefully.
    return 0;
}
