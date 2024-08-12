/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/options/detector_input_options.hpp"
#include "traccc/options/full_tracking_input_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/container_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>

namespace po = boost::program_options;

int seq_run(const traccc::full_tracking_input_config& i_cfg,
            const traccc::common_options& common_opts,
            const traccc::detector_input_options& det_opts, bool run_cpu) {

    // Read the surface transforms
    auto surface_transforms = traccc::io::read_geometry(det_opts.detector_file);

    // Read the digitization configuration file
    auto digi_cfg =
        traccc::io::read_digitization_config(i_cfg.digitization_config_file);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    // uint64_t n_clusters = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_spacepoints_cuda = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_cuda = 0;

    // Configs
    traccc::seedfinder_config finder_config;
    traccc::spacepoint_grid_config grid_config(finder_config);
    traccc::seedfilter_config filter_config;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};

    traccc::clusterization_algorithm ca(host_mr);
    traccc::spacepoint_formation sf(host_mr);
    traccc::seeding_algorithm sa(finder_config, grid_config, filter_config,
                                 host_mr);
    traccc::track_params_estimation tp(host_mr);

    traccc::cuda::stream stream;

    vecmem::cuda::async_copy copy{stream.cudaStream()};

    traccc::cuda::clusterization_algorithm ca_cuda(
        mr, copy, stream, common_opts.target_cells_per_partition);
    traccc::cuda::seeding_algorithm sa_cuda(finder_config, grid_config,
                                            filter_config, mr, copy, stream);
    traccc::cuda::track_params_estimation tp_cuda(mr, copy, stream);

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {
        
        printf("\n\nFor event %d:\n", event);

        const auto WallbeginT = std::chrono::high_resolution_clock::now();

        // Instantiate host containers/collections
        traccc::io::cell_reader_output read_out_per_event(mr.host);
        traccc::clusterization_algorithm::output_type measurements_per_event;
        traccc::spacepoint_formation::output_type spacepoints_per_event;
        traccc::seeding_algorithm::output_type seeds;
        traccc::track_params_estimation::output_type params;

        // Instantiate cuda containers/collections
        traccc::spacepoint_collection_types::buffer spacepoints_cuda_buffer(
            0, *mr.host);
        traccc::seed_collection_types::buffer seeds_cuda_buffer(0, *mr.host);
        traccc::bound_track_parameters_collection_types::buffer
            params_cuda_buffer(0, *mr.host);

        {
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            {
                traccc::performance::timer t("File reading  (cpu)",
                                             elapsedTimes);
                const auto beginT = std::chrono::high_resolution_clock::now();
                // Read the cells from the relevant event file into host memory.
                traccc::io::read_cells(read_out_per_event, event,
                                       common_opts.input_directory,
                                       common_opts.input_data_format,
                                       &surface_transforms, &digi_cfg);
                const auto endT = std::chrono::high_resolution_clock::now();
                std::cout << "Time for file reading: " << std::chrono::duration<double>(endT - beginT).count()*1000 << std::endl;
            }  // stop measuring file reading timer

            const traccc::cell_collection_types::host& cells_per_event =
                read_out_per_event.cells;
            const traccc::cell_module_collection_types::host&
                modules_per_event = read_out_per_event.modules;

            /*-----------------------------
                Clusterization and Spacepoint Creation (cuda)
            -----------------------------*/
            // Create device copy of input collections

            const auto memCopybeginT = std::chrono::high_resolution_clock::now();
            traccc::cell_collection_types::buffer cells_buffer(
                cells_per_event.size(), mr.main);
            copy(vecmem::get_data(cells_per_event), cells_buffer);
            traccc::cell_module_collection_types::buffer modules_buffer(
                modules_per_event.size(), mr.main);
            copy(vecmem::get_data(modules_per_event), modules_buffer);
            const auto memCopyendT = std::chrono::high_resolution_clock::now();
            std::cout << "Time for host to dev mem copy: " << std::chrono::duration<double>(memCopyendT - memCopybeginT).count()*1000 << std::endl;

            {
                {
                    traccc::performance::timer t("Clusterization (cuda)",
                                                elapsedTimes);
                    const auto beginT = std::chrono::high_resolution_clock::now();
                    
                    // Reconstruct it into spacepoints on the device.
                    spacepoints_cuda_buffer =
                        ca_cuda(cells_buffer, modules_buffer).first;
                    stream.synchronize();
                    const auto endT = std::chrono::high_resolution_clock::now();
                    std::cout << "Time for cuda clustering execution: " << std::chrono::duration<double>(endT - beginT).count()*1000 << std::endl;

                }
            }  // stop measuring clusterization cuda timer
            
            // end timing for IO, memcopy and clustering 
            const auto WallendT = std::chrono::high_resolution_clock::now();
            std::cout << "Wall time: " << std::chrono::duration<double>(WallendT - WallbeginT).count()*1000 << std::endl;
            if (run_cpu) {

                /*-----------------------------
                    Clusterization (cpu)
                -----------------------------*/

                {
                    traccc::performance::timer t("Clusterization  (cpu)",
                                                 elapsedTimes);
                    measurements_per_event =
                        ca(cells_per_event, modules_per_event);
                }  // stop measuring clusterization cpu timer

                /*---------------------------------
                    Spacepoint formation (cpu)
                ---------------------------------*/

                {
                    traccc::performance::timer t("Spacepoint formation  (cpu)",
                                                 elapsedTimes);
                    spacepoints_per_event =
                        sf(measurements_per_event, modules_per_event);
                }  // stop measuring spacepoint formation cpu timer
            }

            /*----------------------------
                Seeding algorithm
            ----------------------------*/

            // CUDA

            {
                traccc::performance::timer t("Seeding (cuda)", elapsedTimes);
                seeds_cuda_buffer = sa_cuda(spacepoints_cuda_buffer);
                stream.synchronize();
            }  // stop measuring seeding cuda timer

            // CPU

            if (run_cpu) {
                traccc::performance::timer t("Seeding  (cpu)", elapsedTimes);
                seeds = sa(spacepoints_per_event);
            }  // stop measuring seeding cpu timer

            /*----------------------------
            Track params estimation
            ----------------------------*/

            // CUDA

            {
                traccc::performance::timer t("Track params (cuda)",
                                             elapsedTimes);
                params_cuda_buffer =
                    tp_cuda(spacepoints_cuda_buffer, seeds_cuda_buffer,
                            {0.f, 0.f, finder_config.bFieldInZ});
                stream.synchronize();
            }  // stop measuring track params timer

            // CPU

            if (run_cpu) {
                traccc::performance::timer t("Track params  (cpu)",
                                             elapsedTimes);
                params = tp(spacepoints_per_event, seeds,
                            {0.f, 0.f, finder_config.bFieldInZ});
            }  // stop measuring track params cpu timer

        }  // Stop measuring wall time

        /*----------------------------------
          compare cpu and cuda result
          ----------------------------------*/

        traccc::spacepoint_collection_types::host spacepoints_per_event_cuda;
        traccc::seed_collection_types::host seeds_cuda;
        traccc::bound_track_parameters_collection_types::host params_cuda;

        copy(spacepoints_cuda_buffer, spacepoints_per_event_cuda)->wait();
        copy(seeds_cuda_buffer, seeds_cuda)->wait();
        copy(params_cuda_buffer, params_cuda)->wait();

        if (run_cpu) {

            // Show which event we are currently presenting the results for.
            std::cout << "===>>> Event " << event << " <<<===" << std::endl;

            // Compare the spacepoints made on the host and on the device.
            traccc::collection_comparator<traccc::spacepoint>
                compare_spacepoints{"spacepoints"};
            compare_spacepoints(vecmem::get_data(spacepoints_per_event),
                                vecmem::get_data(spacepoints_per_event_cuda));

            // Compare the seeds made on the host and on the device
            traccc::collection_comparator<traccc::seed> compare_seeds{
                "seeds", traccc::details::comparator_factory<traccc::seed>{
                             vecmem::get_data(spacepoints_per_event),
                             vecmem::get_data(spacepoints_per_event_cuda)}};
            compare_seeds(vecmem::get_data(seeds),
                          vecmem::get_data(seeds_cuda));

            // Compare the track parameters made on the host and on the device.
            traccc::collection_comparator<traccc::bound_track_parameters>
                compare_track_parameters{"track parameters"};
            compare_track_parameters(vecmem::get_data(params),
                                     vecmem::get_data(params_cuda));
        }
        /// Statistics
        n_modules += read_out_per_event.modules.size();
        n_cells += read_out_per_event.cells.size();
        n_measurements += measurements_per_event.size();
        n_spacepoints += spacepoints_per_event.size();
        n_seeds += seeds.size();
        n_spacepoints_cuda += spacepoints_per_event_cuda.size();
        n_seeds_cuda += seeds_cuda.size();

        std::cout << "- created (cuda) " << spacepoints_per_event_cuda.size()
                << " spacepoints     " << std::endl;

        if (common_opts.check_performance) {

            traccc::event_map evt_map(
                event, det_opts.detector_file, i_cfg.digitization_config_file,
                common_opts.input_directory, common_opts.input_directory,
                common_opts.input_directory, host_mr);
            sd_performance_writer.write(
                vecmem::get_data(seeds_cuda),
                vecmem::get_data(spacepoints_per_event_cuda), evt_map);
        }
    }

    if (common_opts.check_performance) {
        sd_performance_writer.finalize();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells from " << n_modules
              << " modules" << std::endl;
    std::cout << "- created (cpu)  " << n_measurements << " measurements     "
              << std::endl;
    std::cout << "- created (cpu)  " << n_spacepoints << " spacepoints     "
              << std::endl;
    std::cout << "- created (cuda) " << n_spacepoints_cuda
              << " spacepoints     " << std::endl;

    std::cout << "- created  (cpu) " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (cuda) " << n_seeds_cuda << " seeds" << std::endl;
    std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    traccc::common_options common_opts(desc);
    traccc::detector_input_options det_opts(desc);
    traccc::full_tracking_input_config full_tracking_input_cfg(desc);
    desc.add_options()("run_cpu", po::value<bool>()->default_value(false),
                       "run cpu tracking as well");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    common_opts.read(vm);
    det_opts.read(vm);
    full_tracking_input_cfg.read(vm);
    auto run_cpu = vm["run_cpu"].as<bool>();

    std::cout << "Running " << argv[0] << " "
              << full_tracking_input_cfg.detector_file << " "
              << common_opts.input_directory << " " << common_opts.events
              << std::endl;

    return seq_run(full_tracking_input_cfg, common_opts, det_opts, run_cpu);
}
