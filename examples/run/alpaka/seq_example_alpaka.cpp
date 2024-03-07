/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"
#include "traccc/alpaka/seeding/seeding_algorithm.hpp"
#include "traccc/alpaka/seeding/track_params_estimation.hpp"
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
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
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>
#endif

#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>

namespace po = boost::program_options;

int seq_run(const traccc::full_tracking_input_options& i_cfg,
            const traccc::common_options& common_opts,
            const traccc::detector_input_options& det_opts, bool run_cpu) {

    // Read the surface transforms
    auto [surface_transforms, barcode_map] = traccc::io::read_geometry(
        det_opts.detector_file,
        (det_opts.use_detray_detector ? traccc::data_format::json
                                      : traccc::data_format::csv));

    // Read the digitization configuration file
    auto digi_cfg =
        traccc::io::read_digitization_config(i_cfg.digitization_config_file);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    // uint64_t n_clusters = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_spacepoints_alpaka = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_alpaka = 0;

    // Configs
    traccc::seedfinder_config finder_config;
    traccc::spacepoint_grid_config grid_config(finder_config);
    traccc::seedfilter_config filter_config;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    vecmem::cuda::copy copy;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};
#else
    vecmem::copy copy;
    traccc::memory_resource mr{host_mr, &host_mr};
#endif

    traccc::clusterization_algorithm ca(host_mr);
    traccc::spacepoint_formation sf(host_mr);
    traccc::seeding_algorithm sa(finder_config, grid_config, filter_config,
                                 host_mr);
    traccc::track_params_estimation tp(host_mr);

    traccc::alpaka::clusterization_algorithm ca_alpaka(
        mr, copy, common_opts.target_cells_per_partition);
    traccc::alpaka::seeding_algorithm sa_alpaka(finder_config, grid_config,
                                                filter_config, mr, copy);
    traccc::alpaka::track_params_estimation tp_alpaka(mr, copy);

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {

        // Instantiate host containers/collections
        traccc::io::cell_reader_output read_out_per_event(mr.host);
        traccc::clusterization_algorithm::output_type measurements_per_event;
        traccc::spacepoint_formation::output_type spacepoints_per_event;
        traccc::seeding_algorithm::output_type seeds;
        traccc::track_params_estimation::output_type params;

        // Instantiate alpaka containers/collections
        traccc::spacepoint_collection_types::buffer spacepoints_alpaka_buffer(
            0, *mr.host);
        traccc::seed_collection_types::buffer seeds_alpaka_buffer(0, *mr.host);
        traccc::bound_track_parameters_collection_types::buffer
            params_alpaka_buffer(0, *mr.host);

        {
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            {
                traccc::performance::timer t("File reading  (cpu)",
                                             elapsedTimes);
                // Read the cells from the relevant event file into host memory.
                traccc::io::read_cells(
                    read_out_per_event, event, common_opts.input_directory,
                    common_opts.input_data_format, &surface_transforms,
                    &digi_cfg, barcode_map.get());
            }  // stop measuring file reading timer

            const traccc::cell_collection_types::host& cells_per_event =
                read_out_per_event.cells;
            const traccc::cell_module_collection_types::host&
                modules_per_event = read_out_per_event.modules;

            /*-----------------------------
                Clusterization and Spacepoint Creation (alpaka)
            -----------------------------*/
            // Create device copy of input collections
            traccc::cell_collection_types::buffer cells_buffer(
                cells_per_event.size(), mr.main);
            copy(vecmem::get_data(cells_per_event), cells_buffer);
            traccc::cell_module_collection_types::buffer modules_buffer(
                modules_per_event.size(), mr.main);
            copy(vecmem::get_data(modules_per_event), modules_buffer);

            {
                traccc::performance::timer t("Clusterization (alpaka)",
                                             elapsedTimes);
                // Reconstruct it into spacepoints on the device.
                spacepoints_alpaka_buffer =
                    ca_alpaka(cells_buffer, modules_buffer).first;
            }  // stop measuring clusterization alpaka timer

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

            // Alpaka

            {
                traccc::performance::timer t("Seeding (alpaka)", elapsedTimes);
                seeds_alpaka_buffer = sa_alpaka(spacepoints_alpaka_buffer);
            }  // stop measuring seeding alpaka timer

            // CPU

            if (run_cpu) {
                traccc::performance::timer t("Seeding  (cpu)", elapsedTimes);
                seeds = sa(spacepoints_per_event);
            }  // stop measuring seeding cpu timer

            /*----------------------------
            Track params estimation
            ----------------------------*/

            // Alpaka

            {
                traccc::performance::timer t("Track params (alpaka)",
                                             elapsedTimes);
                params_alpaka_buffer =
                    tp_alpaka(spacepoints_alpaka_buffer, seeds_alpaka_buffer,
                              {0.f, 0.f, finder_config.bFieldInZ});
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
          compare cpu and alpaka result
          ----------------------------------*/

        traccc::spacepoint_collection_types::host spacepoints_per_event_alpaka;
        traccc::seed_collection_types::host seeds_alpaka;
        traccc::bound_track_parameters_collection_types::host params_alpaka;
        copy(spacepoints_alpaka_buffer, spacepoints_per_event_alpaka)->wait();
        copy(seeds_alpaka_buffer, seeds_alpaka)->wait();
        copy(params_alpaka_buffer, params_alpaka)->wait();

        if (run_cpu) {

            // Show which event we are currently presenting the results for.
            std::cout << "===>>> Event " << event << " <<<===" << std::endl;

            // Compare the spacepoints made on the host and on the device.
            traccc::collection_comparator<traccc::spacepoint>
                compare_spacepoints{"spacepoints"};
            compare_spacepoints(vecmem::get_data(spacepoints_per_event),
                                vecmem::get_data(spacepoints_per_event_alpaka));

            // Compare the seeds made on the host and on the device
            traccc::collection_comparator<traccc::seed> compare_seeds{
                "seeds", traccc::details::comparator_factory<traccc::seed>{
                             vecmem::get_data(spacepoints_per_event),
                             vecmem::get_data(spacepoints_per_event_alpaka)}};
            compare_seeds(vecmem::get_data(seeds),
                          vecmem::get_data(seeds_alpaka));

            // Compare the track parameters made on the host and on the device.
            traccc::collection_comparator<traccc::bound_track_parameters>
                compare_track_parameters{"track parameters"};
            compare_track_parameters(vecmem::get_data(params),
                                     vecmem::get_data(params_alpaka));
        }

        /// Statistics
        n_modules += read_out_per_event.modules.size();
        n_cells += read_out_per_event.cells.size();
        n_measurements += measurements_per_event.size();
        n_spacepoints += spacepoints_per_event.size();
        n_seeds += seeds.size();
        n_spacepoints_alpaka += spacepoints_per_event_alpaka.size();
        n_seeds_alpaka += seeds_alpaka.size();

        if (common_opts.check_performance) {

            traccc::event_map evt_map(
                event, det_opts.detector_file, i_cfg.digitization_config_file,
                common_opts.input_directory, common_opts.input_directory,
                common_opts.input_directory, host_mr);
            sd_performance_writer.write(
                vecmem::get_data(seeds_alpaka),
                vecmem::get_data(spacepoints_per_event_alpaka), evt_map);
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
    std::cout << "- created (alpaka) " << n_spacepoints_alpaka
              << " spacepoints     " << std::endl;

    std::cout << "- created  (cpu) " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (alpaka) " << n_seeds_alpaka << " seeds"
              << std::endl;
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
    traccc::full_tracking_input_options full_tracking_input_cfg(desc);
    desc.add_options()("run-cpu", po::value<bool>()->default_value(false),
                       "run cpu tracking as well");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    common_opts.read(vm);
    det_opts.read(vm);
    full_tracking_input_cfg.read(vm);
    auto run_cpu = vm["run-cpu"].as<bool>();

    // Tell the user what's happening.
    std::cout << "\nRunning the full tracking chain using Alpaka\n\n"
              << common_opts << "\n"
              << det_opts << "\n"
              << full_tracking_input_cfg << "\n"
              << std::endl;

    return seq_run(full_tracking_input_cfg, common_opts, det_opts, run_cpu);
}
