/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// io
#include "traccc/io/read.hpp"

// algorithms
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"

// performance
#include "traccc/performance/throughput.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/performance/timing_info.hpp"

// options
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/throughput_options.hpp"

// Vecmem include(s)
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// System include(s).
#include <cstdlib>
#include <ctime>
#include <exception>
#include <iostream>

namespace po = boost::program_options;

int throughput_seq_run(const traccc::throughput_options& i_cfg) {

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &host_mr};

    vecmem::cuda::copy copy;

    traccc::device::container_h2d_copy_alg<traccc::cell_container_types>
        cell_h2d{mr, copy};
    traccc::cuda::clusterization_algorithm ca(mr);
    traccc::cuda::seeding_algorithm sa(mr);
    traccc::cuda::track_params_estimation tp(mr);
    traccc::bound_track_parameters_collection_types::host params_copy;

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
    std::size_t rec_track_params = 0;

    // Cold Run events
    // These are not accounted for the performance measurement
    for (std::size_t i = 0; i < i_cfg.cold_run_events; ++i) {
        const int event = std::rand() % i_cfg.loaded_events;
        traccc::cell_container_types::host& cells_per_event =
            cells_event_vec[event];
        const traccc::cell_container_types::buffer cells_buffer =
            cell_h2d(traccc::get_data(cells_per_event));
        auto spacepoints_buffer = ca(cells_buffer);
        auto seeds_buffer = sa(spacepoints_buffer);
        auto params_buffer = tp(spacepoints_buffer, seeds_buffer);
        copy(params_buffer, params_copy);
        rec_track_params += params_copy.size();
    }

    // Loop over events
    for (std::size_t i = 0; i < i_cfg.processed_events; ++i) {

        const std::size_t event = std::rand() % i_cfg.loaded_events;

        {

            traccc::performance::timer t("Event processing", elapsedTimes);

            traccc::cell_container_types::host& cells_per_event =
                cells_event_vec[event];

            // Copy the cell data to the device.
            const traccc::cell_container_types::buffer cells_buffer =
                cell_h2d(traccc::get_data(cells_per_event));

            /*-------------------------------------------
                Clusterization & Spacepoint formation
            -------------------------------------------*/

            auto spacepoints_buffer = ca(cells_buffer);

            /*-----------------------
              Seeding algorithm
              -----------------------*/

            auto seeds_buffer = sa(spacepoints_buffer);

            /*----------------------------
              Track params estimation
              ----------------------------*/

            auto params_buffer = tp(spacepoints_buffer, seeds_buffer);

            copy(params_buffer, params_copy);
            rec_track_params += params_copy.size();
        }
    }
    std::cout << "Reconstructed track parameters: " << rec_track_params
              << std::endl;
    std::cout << "Time totals:" << std::endl;
    std::cout << elapsedTimes << std::endl;
    std::cout << "Throughput:" << std::endl;
    std::cout << traccc::performance::throughput{i_cfg.processed_events,
                                                 elapsedTimes,
                                                 "Event processing"}
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
    traccc::throughput_options cfg(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    cfg.read(vm);

    std::cout << "\n Cuda throughput test with single-threaded CPU\n\n"
              << cfg << "\n"
              << std::endl;

    return throughput_seq_run(cfg);
}
