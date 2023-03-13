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
#include "traccc/io/read_cells_alt.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/options/common_options.hpp"
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
#include <vecmem/utils/cuda/copy.hpp>

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>

namespace po = boost::program_options;

int seq_run(const traccc::full_tracking_input_config& i_cfg,
            const traccc::common_options& common_opts, bool run_cpu) {

    // Read the surface transforms
    auto surface_transforms = traccc::io::read_geometry(i_cfg.detector_file);

    // Read the digitization configuration file
    auto digi_cfg =
        traccc::io::read_digitization_config(i_cfg.digitization_config_file);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_alt_cells = 0;
    uint64_t n_modules = 0;
    // uint64_t n_clusters = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_spacepoints_cuda = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_cuda = 0;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};

    traccc::clusterization_algorithm ca(host_mr);
    traccc::spacepoint_formation sf(host_mr);
    traccc::seeding_algorithm sa(host_mr);
    traccc::track_params_estimation tp(host_mr);

    traccc::cuda::stream stream;

    vecmem::cuda::copy copy;
    vecmem::cuda::async_copy async_copy{stream.cudaStream()};

    traccc::cuda::clusterization_algorithm ca_cuda(
        mr, async_copy, stream, common_opts.target_cells_per_partition);
    traccc::cuda::seeding_algorithm sa_cuda(mr, async_copy, stream);
    traccc::cuda::track_params_estimation tp_cuda(mr, async_copy, stream);

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});
    if (i_cfg.check_performance) {
        sd_performance_writer.add_cache("CPU");
        sd_performance_writer.add_cache("CUDA");
    }

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {

        // Instantiate host containers/collections
        traccc::cell_container_types::host cells_per_event;
        traccc::alt_cell_reader_output_t alt_read_out_per_event;
        traccc::clusterization_algorithm::output_type measurements_per_event;
        traccc::spacepoint_formation::output_type spacepoints_per_event;
        traccc::seeding_algorithm::output_type seeds;
        traccc::track_params_estimation::output_type params;

        // Instantiate cuda containers/collections
        traccc::spacepoint_collection_types::buffer spacepoints_cuda_buffer(
            0, *mr.host);
        traccc::alt_seed_collection_types::buffer seeds_cuda_buffer(0,
                                                                    *mr.host);
        traccc::bound_track_parameters_collection_types::buffer
            params_cuda_buffer(0, *mr.host);

        {
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            if (run_cpu) {
                traccc::performance::timer t("File reading  (cpu)",
                                             elapsedTimes);
                // Read the cells from the relevant event file into host memory.
                cells_per_event = traccc::io::read_cells(
                    event, common_opts.input_directory,
                    common_opts.input_data_format, &surface_transforms,
                    &digi_cfg, &cuda_host_mr);
            }  // stop measuring file reading timer

            {
                traccc::performance::timer t("Alt File reading  (cpu)",
                                             elapsedTimes);
                // Read the cells from the relevant event file into host memory.
                alt_read_out_per_event = traccc::io::read_cells_alt(
                    event, common_opts.input_directory,
                    common_opts.input_data_format, &surface_transforms,
                    &digi_cfg, &cuda_host_mr);
            }  // stop measuring file reading timer

            const traccc::alt_cell_collection_types::host& alt_cells_per_event =
                alt_read_out_per_event.cells;
            const traccc::cell_module_collection_types::host&
                modules_per_event = alt_read_out_per_event.modules;

            /*-----------------------------
                Clusterization and Spacepoint Creation (cuda)
            -----------------------------*/
            // Create device copy of input collections
            traccc::alt_cell_collection_types::buffer cells_buffer(
                alt_cells_per_event.size(), mr.main);
            copy(vecmem::get_data(alt_cells_per_event), cells_buffer);
            traccc::cell_module_collection_types::buffer modules_buffer(
                modules_per_event.size(), mr.main);
            copy(vecmem::get_data(modules_per_event), modules_buffer);

            {
                traccc::performance::timer t("Clusterization (cuda)",
                                             elapsedTimes);
                // Reconstruct it into spacepoints on the device.
                spacepoints_cuda_buffer =
                    ca_cuda(cells_buffer, modules_buffer).first;
                stream.synchronize();
            }  // stop measuring clusterization cuda timer

            if (run_cpu) {

                /*-----------------------------
                    Clusterization (cpu)
                -----------------------------*/

                {
                    traccc::performance::timer t("Clusterization  (cpu)",
                                                 elapsedTimes);
                    measurements_per_event = ca(cells_per_event);
                }  // stop measuring clusterization cpu timer

                /*---------------------------------
                    Spacepoint formation (cpu)
                ---------------------------------*/

                {
                    traccc::performance::timer t("Spacepoint formation  (cpu)",
                                                 elapsedTimes);
                    spacepoints_per_event = sf(measurements_per_event);
                }  // stop measuring spacepoint formation cpu timer
            }

            /*----------------------------
                Seeding algorithm
            ----------------------------*/

            // CUDA

            {
                traccc::performance::timer t("Seeding (cuda)", elapsedTimes);
                seeds_cuda_buffer = sa_cuda(spacepoints_cuda_buffer);
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
                    tp_cuda(spacepoints_cuda_buffer, seeds_cuda_buffer);
            }  // stop measuring track params timer

            // CPU

            if (run_cpu) {
                traccc::performance::timer t("Track params  (cpu)",
                                             elapsedTimes);
                params = tp(spacepoints_per_event, seeds);
            }  // stop measuring track params cpu timer

        }  // Stop measuring wall time

        /*----------------------------------
          compare cpu and cuda result
          ----------------------------------*/

        traccc::spacepoint_collection_types::host spacepoints_per_event_cuda;
        traccc::alt_seed_collection_types::host seeds_cuda;
        traccc::bound_track_parameters_collection_types::host params_cuda;
        if (run_cpu || i_cfg.check_performance) {
            copy(spacepoints_cuda_buffer, spacepoints_per_event_cuda);
            copy(seeds_cuda_buffer, seeds_cuda);
            copy(params_cuda_buffer, params_cuda);
        }

        if (run_cpu) {

            // Show which event we are currently presenting the results for.
            std::cout << "===>>> Event " << event << " <<<===" << std::endl;

            /// TODO:
            // Compare the spacepoints made on the host and on the device.
            /* traccc::container_comparator<traccc::geometry_id,
                                         traccc::spacepoint>
                compare_spacepoints{"spacepoints"};
            compare_spacepoints(traccc::get_data(spacepoints_per_event),
                                traccc::get_data(spacepoints_per_event_cuda));

            // Compare the seeds made on the host and on the device
            traccc::collection_comparator<traccc::seed> compare_seeds{
                "seeds", traccc::details::comparator_factory<traccc::seed>{
                             traccc::get_data(spacepoints_per_event),
                             traccc::get_data(spacepoints_per_event_cuda)}};
            compare_seeds(vecmem::get_data(seeds),
                          vecmem::get_data(seeds_cuda)); */

            // Compare the track parameters made on the host and on the device.
            traccc::collection_comparator<traccc::bound_track_parameters>
                compare_track_parameters{"track parameters"};
            compare_track_parameters(vecmem::get_data(params),
                                     vecmem::get_data(params_cuda));

            /// Statistics
            n_modules += cells_per_event.size();
            n_cells += cells_per_event.total_size();
            n_alt_cells += alt_read_out_per_event.cells.size();
            n_measurements += measurements_per_event.total_size();
            n_spacepoints += spacepoints_per_event.total_size();
            n_spacepoints_cuda += spacepoints_per_event_cuda.size();
            n_seeds_cuda += seeds_cuda.size();
            n_seeds += seeds.size();
        }
        /// TODO:
        /* if (i_cfg.check_performance) {

            traccc::event_map evt_map(
                event, i_cfg.detector_file,
                i_cfg.digitization_config_file,
                common_opts.input_directory, common_opts.input_directory,
                common_opts.input_directory, host_mr);
            sd_performance_writer.write("CUDA", seeds_cuda,
                                        spacepoints_per_event_cuda,
                                        evt_map);

            if (run_cpu) {
                sd_performance_writer.write("CPU", seeds,
                spacepoints_per_event,
                                            evt_map);
            }
        } */
    }

    /* if (i_cfg.check_performance) {
        sd_performance_writer.finalize();
    } */

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints from "
              << n_modules << " modules" << std::endl;
    std::cout << "- created        " << n_cells << " cells           "
              << std::endl;
    std::cout << "- created        " << n_alt_cells << " alt cells"
              << std::endl;
    std::cout << "- created        " << n_measurements << " meaurements     "
              << std::endl;
    std::cout << "- created        " << n_spacepoints << " spacepoints     "
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
    traccc::full_tracking_input_config full_tracking_input_cfg(desc);
    desc.add_options()("run_cpu", po::value<bool>()->default_value(false),
                       "run cpu tracking as well");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    common_opts.read(vm);
    full_tracking_input_cfg.read(vm);
    auto run_cpu = vm["run_cpu"].as<bool>();

    std::cout << "Running " << argv[0] << " "
              << full_tracking_input_cfg.detector_file << " "
              << common_opts.input_directory << " " << common_opts.events
              << std::endl;

    return seq_run(full_tracking_input_cfg, common_opts, run_cpu);
}
