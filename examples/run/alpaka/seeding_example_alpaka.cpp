/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/alpaka/seeding/seeding_algorithm.hpp"
#include "traccc/alpaka/seeding/track_params_estimation.hpp"
#include "traccc/definitions/common.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/efficiency/nseed_performance_writer.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/efficiency/track_filter.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/options/detector_input_options.hpp"
#include "traccc/options/finding_input_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/propagation_options.hpp"
#include "traccc/options/seeding_input_options.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/core/detector_metadata.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#include <vecmem/memory/hip/device_memory_resource.hpp>
#include <vecmem/memory/hip/host_memory_resource.hpp>
#include <vecmem/memory/hip/managed_memory_resource.hpp>
#include <vecmem/utils/hip/copy.hpp>
#endif

#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>

using namespace traccc;
namespace po = boost::program_options;

int seq_run(const traccc::seeding_input_options& /*i_cfg*/,
            const traccc::finding_input_options& /*finding_cfg*/,
            const traccc::propagation_options& /*propagation_opts*/,
            const traccc::common_options& common_opts,
            const traccc::detector_input_options& det_opts, bool run_cpu) {

    using host_detector_type = detray::detector<>;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    vecmem::cuda::copy copy;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    vecmem::cuda::managed_memory_resource mng_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    vecmem::copy copy;
    vecmem::host_memory_resource mng_mr;
    traccc::memory_resource mr{host_mr, &host_mr};
#endif

    // Performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});

    traccc::nseed_performance_writer nsd_performance_writer(
        "nseed_performance_",
        std::make_unique<traccc::simple_charged_eta_pt_cut>(
            2.7f, 1.f * traccc::unit<traccc::scalar>::GeV),
        std::make_unique<traccc::stepped_percentage>(0.6f));

    if (common_opts.check_performance) {
        nsd_performance_writer.initialize();
    }

    // Output stats
    uint64_t n_modules = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_alpaka = 0;

    /*****************************
     * Build a geometry
     *****************************/

    // Read the detector
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(traccc::io::data_directory() + det_opts.detector_file);
    if (!det_opts.material_file.empty()) {
        reader_cfg.add_file(traccc::io::data_directory() +
                            det_opts.material_file);
    }
    if (!det_opts.grid_file.empty()) {
        reader_cfg.add_file(traccc::io::data_directory() + det_opts.grid_file);
    }
    auto [host_det, names] =
        detray::io::read_detector<host_detector_type>(mng_mr, reader_cfg);

    traccc::geometry surface_transforms =
        traccc::io::alt_read_geometry(host_det);

    // Seeding algorithms
    traccc::seedfinder_config finder_config;
    traccc::spacepoint_grid_config grid_config(finder_config);
    traccc::seedfilter_config filter_config;

    traccc::seeding_algorithm sa(finder_config, grid_config, filter_config,
                                 host_mr);
    traccc::track_params_estimation tp(host_mr);

    // Alpaka Algorithms
    traccc::alpaka::seeding_algorithm sa_alpaka{finder_config, grid_config,
                                                filter_config, mr, copy};
    traccc::alpaka::track_params_estimation tp_alpaka{mr, copy};

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {

        // Instantiate host containers/collections
        traccc::io::spacepoint_reader_output sp_reader_output(mr.host);
        traccc::io::measurement_reader_output meas_reader_output(mr.host);

        traccc::seeding_algorithm::output_type seeds;
        traccc::track_params_estimation::output_type params;

        // Instantiate alpaka containers/collections
        traccc::seed_collection_types::buffer seeds_alpaka_buffer(0,
                                                                  *(mr.host));
        traccc::bound_track_parameters_collection_types::buffer
            params_alpaka_buffer(0, *mr.host);

        {  // Start measuring wall time
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            /*-----------------
            hit file reading
            -----------------*/
            {
                traccc::performance::timer t("Hit reading  (cpu)",
                                             elapsedTimes);
                // Read the hits from the relevant event file
                traccc::io::read_spacepoints(
                    sp_reader_output, event, common_opts.input_directory,
                    surface_transforms, common_opts.input_data_format);

                // Read measurements
                traccc::io::read_measurements(meas_reader_output, event,
                                              common_opts.input_directory,
                                              common_opts.input_data_format);
            }  // stop measuring hit reading timer

            auto& spacepoints_per_event = sp_reader_output.spacepoints;
            auto& modules_per_event = sp_reader_output.modules;
            // auto& measurements_per_event = meas_reader_output.measurements;

            /*----------------------------
                Seeding algorithm
            ----------------------------*/

            // Alpaka

            // TODO: Check this (and all other copies) are intelligent.
            // Copy the spacepoint data to the device.
            traccc::spacepoint_collection_types::buffer
                spacepoints_alpaka_buffer(spacepoints_per_event.size(),
                                          mr.main);
            copy(vecmem::get_data(spacepoints_per_event),
                 spacepoints_alpaka_buffer);
            traccc::cell_module_collection_types::buffer modules_buffer(
                modules_per_event.size(), mr.main);
            copy(vecmem::get_data(modules_per_event), modules_buffer);

            {
                traccc::performance::timer t("Seeding (alpaka)", elapsedTimes);
                // Reconstruct the spacepoints into seeds.
                seeds_alpaka_buffer =
                    sa_alpaka(vecmem::get_data(spacepoints_alpaka_buffer));
            }

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
                params_alpaka_buffer = tp_alpaka(
                    spacepoints_alpaka_buffer, seeds_alpaka_buffer,
                    modules_buffer, {0.f, 0.f, finder_config.bFieldInZ});
            }  // stop measuring track params alpaka timer
            // CPU

            if (run_cpu) {
                traccc::performance::timer t("Track params  (cpu)",
                                             elapsedTimes);
                params = tp(std::move(spacepoints_per_event), seeds,
                            {0.f, 0.f, finder_config.bFieldInZ});
            }  // stop measuring track params cpu timer

        }  // Stop measuring wall time

        /*----------------------------------
          compare seeds from cpu and alpaka
          ----------------------------------*/

        // Copy the seeds to the host for comparisons
        traccc::seed_collection_types::host seeds_alpaka;
        traccc::bound_track_parameters_collection_types::host params_alpaka;
        copy(seeds_alpaka_buffer, seeds_alpaka);
        copy(params_alpaka_buffer, params_alpaka);

        if (run_cpu) {
            // Show which event we are currently presenting the results for.
            std::cout << "===>>> Event " << event << " <<<===" << std::endl;

            // Compare the seeds made on the host and on the device
            traccc::collection_comparator<traccc::seed> compare_seeds{
                "seeds", traccc::details::comparator_factory<traccc::seed>{
                             vecmem::get_data(sp_reader_output.spacepoints),
                             vecmem::get_data(sp_reader_output.spacepoints)}};
            compare_seeds(vecmem::get_data(seeds),
                          vecmem::get_data(seeds_alpaka));

            // Compare the track parameters made on the host and on the device.
            traccc::collection_comparator<traccc::bound_track_parameters>
                compare_track_parameters{"track parameters"};
            compare_track_parameters(vecmem::get_data(params),
                                     vecmem::get_data(params_alpaka));
        }

        /*----------------
             Statistics
          ---------------*/

        n_spacepoints += sp_reader_output.spacepoints.size();
        n_modules += sp_reader_output.modules.size();
        n_seeds_alpaka += seeds_alpaka.size();
        n_seeds += seeds.size();

        /*------------
          Writer
          ------------*/

        if (common_opts.check_performance) {
            traccc::event_map2 evt_map(event, common_opts.input_directory,
                                       common_opts.input_directory,
                                       common_opts.input_directory);

            sd_performance_writer.write(
                vecmem::get_data(seeds_alpaka),
                vecmem::get_data(sp_reader_output.spacepoints), evt_map);
        }
    }

    if (common_opts.check_performance) {
        sd_performance_writer.finalize();
        nsd_performance_writer.finalize();

        std::cout << nsd_performance_writer.generate_report_str();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints from "
              << n_modules << " modules" << std::endl;
    std::cout << "- created  (cpu)  " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (alpaka)  " << n_seeds_alpaka << " seeds"
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
    traccc::seeding_input_options seeding_input_cfg(desc);
    traccc::finding_input_options finding_input_cfg(desc);
    traccc::propagation_options propagation_opts(desc);
    desc.add_options()("run-cpu", po::value<bool>()->default_value(false),
                       "run cpu tracking as well");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    common_opts.read(vm);
    det_opts.read(vm);
    seeding_input_cfg.read(vm);
    finding_input_cfg.read(vm);
    propagation_opts.read(vm);
    auto run_cpu = vm["run-cpu"].as<bool>();

    // Tell the user what's happening.
    std::cout << "\nRunning the tracking chain using Alpaka\n\n"
              << common_opts << "\n"
              << det_opts << "\n"
              << seeding_input_cfg << "\n"
              << finding_input_cfg << "\n"
              << propagation_opts << "\n"
              << std::endl;

    return seq_run(seeding_input_cfg, finding_input_cfg, propagation_opts,
                   common_opts, det_opts, run_cpu);
}
