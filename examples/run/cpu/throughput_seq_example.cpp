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
#include "traccc/performance/timer.hpp"

// options
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/throughput_full_tracking_input_options.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <time.h>

#include <cstdlib>
#include <exception>
#include <iostream>

namespace po = boost::program_options;

int throughput_seq_run(
    const traccc::throughput_full_tracking_input_config& i_cfg) {

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    traccc::clusterization_algorithm ca(host_mr);
    traccc::spacepoint_formation sf(host_mr);
    traccc::seeding_algorithm sa(host_mr);
    traccc::track_params_estimation tp(host_mr);

    traccc::demonstrator_input cells_event_vec;
    cells_event_vec.reserve(i_cfg.loaded_events);

    traccc::performance::timing_info elapsedTimes;

    {

        traccc::performance::timer t("File reading", elapsedTimes);

        cells_event_vec = traccc::io::read(
            i_cfg.loaded_events, i_cfg.input_directory, i_cfg.detector_file,
            i_cfg.digitization_config_file, i_cfg.input_data_format, &host_mr);
    }
    // Seed random number generator
    std::srand(std::time(0));

    // Dummy count uses output of tp algorithm to ensure the compiler
    // optimisations don't skip this step
    std::size_t dummy_count = 0;

    // Loop over events
    for (int i = 0; i < i_cfg.processed_events; ++i) {

        const int event = std::rand() % i_cfg.loaded_events;

        {

            traccc::performance::timer t("Events processing", elapsedTimes);

            traccc::cell_container_types::host& cells_per_event =
                cells_event_vec[event];

            /*-------------------
                Clusterization
              -------------------*/

            auto measurements_per_event = ca(cells_per_event);

            /*------------------------
                Spacepoint formation
              ------------------------*/

            auto spacepoints_per_event = sf(measurements_per_event);

            /*-----------------------
              Seeding algorithm
              -----------------------*/

            auto seeds = sa(spacepoints_per_event);

            /*----------------------------
              Track params estimation
              ----------------------------*/

            dummy_count += tp(spacepoints_per_event, seeds).size();
        }
    }
    std::cout << "dummy_count = " << dummy_count << std::endl;

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- Processed " << i_cfg.processed_events << " events from "
              << i_cfg.loaded_events << " loaded events." << std::endl;

    std::cout << "File reading throughput (ms/event) = "
              << elapsedTimes.get_time("File reading").count() * 1.e-6 /
                     (double)i_cfg.loaded_events
              << std::endl;
    std::cout << "Event processing        (ms/event) = "
              << elapsedTimes.get_time("Events processing").count() * 1.e-6 /
                     (double)i_cfg.processed_events
              << std::endl;
    std::cout << "Processing throughput   (event/s) = "
              << (double)i_cfg.processed_events /
                     elapsedTimes.get_time("Events processing").count() * 1.e9
              << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    traccc::throughput_full_tracking_input_config cfg(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    cfg.read(vm);

    std::cout << "Running " << argv[0] << " " << cfg << std::endl;

    return throughput_seq_run(cfg);
}
