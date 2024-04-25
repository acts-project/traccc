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
#include "traccc/options/accelerator.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"
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

int seq_run(const traccc::opts::track_seeding& seeding_opts,
            const traccc::opts::track_finding& /*finding_opts*/,
            const traccc::opts::track_propagation& propagation_opts,
            const traccc::opts::input_data& input_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::performance& performance_opts,
            const traccc::opts::accelerator& accelerator_opts) {

    /// Type declarations
    using host_detector_type = detray::detector<>;
    using device_detector_type =
        detray::detector<detray::default_metadata,
                         detray::device_container_types>;

    using b_field_t = covfie::field<detray::bfield::const_bknd_t>;
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t,
                           typename host_detector_type::transform3,
                           detray::constrained_step<>>;
    using host_navigator_type = detray::navigator<const host_detector_type>;
    using host_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, host_navigator_type>;
    using device_navigator_type = detray::navigator<const device_detector_type>;
    using device_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, device_navigator_type>;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    vecmem::cuda::copy copy;
    vecmem::cuda::host_memory_resource host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    vecmem::cuda::managed_memory_resource mng_mr;
    traccc::memory_resource mr{device_mr, &host_mr};
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    vecmem::hip::copy copy;
    vecmem::hip::host_memory_resource host_mr;
    vecmem::hip::device_memory_resource device_mr;
    vecmem::hip::managed_memory_resource mng_mr;
    traccc::memory_resource mr{device_mr, &host_mr};
#else
    vecmem::copy copy;
    vecmem::host_memory_resource host_mr;
    vecmem::host_memory_resource mng_mr;
    traccc::memory_resource mr{host_mr, &host_mr};
#endif

    // Performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});
    // traccc::finding_performance_writer find_performance_writer(
    //     traccc::finding_performance_writer::config{});
    // traccc::fitting_performance_writer fit_performance_writer(
    //     traccc::fitting_performance_writer::config{});

    traccc::nseed_performance_writer nsd_performance_writer(
        "nseed_performance_",
        std::make_unique<traccc::simple_charged_eta_pt_cut>(
            2.7f, 1.f * traccc::unit<traccc::scalar>::GeV),
        std::make_unique<traccc::stepped_percentage>(0.6f));

    if (performance_opts.run) {
        nsd_performance_writer.initialize();
    }

    // Output stats
    uint64_t n_modules = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_alpaka = 0;
    uint64_t n_found_tracks = 0;
    // uint64_t n_found_tracks_alpaka = 0;
    uint64_t n_fitted_tracks = 0;
    // uint64_t n_fitted_tracks_alpaka = 0;

    /*****************************
     * Build a geometry
     *****************************/


    // B field value and its type
    // @TODO: Set B field as argument
    const traccc::vector3 B{0, 0, 2 * detray::unit<traccc::scalar>::T};
    auto field = detray::bfield::create_const_field(B);

    // Read the detector
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(traccc::io::data_directory() +
                        detector_opts.detector_file);
    if (!detector_opts.material_file.empty()) {
        reader_cfg.add_file(traccc::io::data_directory() +
                            detector_opts.material_file);
    }
    if (!detector_opts.grid_file.empty()) {
        reader_cfg.add_file(traccc::io::data_directory() +
                            detector_opts.grid_file);
    }
    auto [host_det, names] =
        detray::io::read_detector<host_detector_type>(mng_mr, reader_cfg);

    traccc::geometry surface_transforms =
        traccc::io::alt_read_geometry(host_det);

    // Detector view object
    auto det_view = detray::get_data(host_det);

    // Copy objects
    // traccc::device::container_d2h_copy_alg<
    //     traccc::track_candidate_container_types>
    //     track_candidate_d2h{mr, copy};

    // traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
    //     track_state_d2h{mr, copy};

    // Seeding algorithms
    traccc::seeding_algorithm sa(seeding_opts.seedfinder,
                                 {seeding_opts.seedfinder},
                                 seeding_opts.seedfilter, host_mr);
    traccc::track_params_estimation tp(host_mr);

    // Alpaka Algorithms
    traccc::alpaka::seeding_algorithm sa_alpaka{seeding_opts.seedfinder,
                                                {seeding_opts.seedfinder},
                                                seeding_opts.seedfilter,
                                                mr,
                                                copy};
    traccc::alpaka::track_params_estimation tp_alpaka{mr, copy};

    // Finding algorithm configuration
    // typename traccc::cuda::finding_algorithm<
    //     rk_stepper_type, device_navigator_type>::config_type cfg;
    // cfg.min_track_candidates_per_track = finding_opts.track_candidates_range[0];
    // cfg.max_track_candidates_per_track = finding_opts.track_candidates_range[1];
    // cfg.chi2_max = finding_opts.chi2_max;
    // propagation_opts.setup(cfg.propagation);

    // // Finding algorithm object
    // traccc::finding_algorithm<rk_stepper_type, host_navigator_type>
    //     host_finding(cfg);
    // traccc::alpaka::finding_algorithm<rk_stepper_type, device_navigator_type>
    //     device_finding(cfg, mr, async_copy, stream);

    // Fitting algorithm object
    typename traccc::fitting_algorithm<host_fitter_type>::config_type fit_cfg;
    propagation_opts.setup(fit_cfg.propagation);

    traccc::fitting_algorithm<host_fitter_type> host_fitting(fit_cfg);
    // traccc::cuda::fitting_algorithm<device_fitter_type> device_fitting(
    //     fit_cfg, mr, async_copy, stream);

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Instantiate host containers/collections
        traccc::io::spacepoint_reader_output sp_reader_output(mr.host);
        traccc::io::measurement_reader_output meas_reader_output(mr.host);

        traccc::seeding_algorithm::output_type seeds;
        traccc::track_params_estimation::output_type params;
        traccc::track_candidate_container_types::host track_candidates;
        traccc::track_state_container_types::host track_states;

        traccc::seed_collection_types::buffer seeds_alpaka_buffer(0, *(mr.host));
        traccc::bound_track_parameters_collection_types::buffer
            params_alpaka_buffer(0, *mr.host);

        // traccc::track_candidate_container_types::buffer
        //     track_candidates_alpaka_buffer{{{}, *(mr.host)},
        //                                  {{}, *(mr.host), mr.host}};

        // traccc::track_state_container_types::buffer track_states_alpaka_buffer{
        //     {{}, *(mr.host)}, {{}, *(mr.host), mr.host}};

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
                    sp_reader_output, event, input_opts.directory,
                    surface_transforms, input_opts.format);

                // Read measurements
                traccc::io::read_measurements(meas_reader_output, event,
                                              input_opts.directory,
                                              input_opts.format);
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

            if (accelerator_opts.compare_with_cpu) {
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
                              {0.f, 0.f, seeding_opts.seedfinder.bFieldInZ});
            }  // stop measuring track params alpaka timer

            // CPU
            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Track params  (cpu)",
                                             elapsedTimes);
                params = tp(std::move(spacepoints_per_event), seeds,
                            {0.f, 0.f, seeding_opts.seedfinder.bFieldInZ});
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

        if (accelerator_opts.compare_with_cpu) {
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

        if (performance_opts.run) {
            traccc::event_map2 evt_map(event, input_opts.directory,
                                       input_opts.directory,
                                       input_opts.directory);

            sd_performance_writer.write(
                vecmem::get_data(seeds_alpaka),
                vecmem::get_data(sp_reader_output.spacepoints), evt_map);
        }
    }

    if (performance_opts.run) {
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

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::accelerator accelerator_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain Using Alpaka (without clusterization)",
        {detector_opts, input_opts, seeding_opts, finding_opts,
         propagation_opts, performance_opts, accelerator_opts},
        argc,
        argv};

    // Run the application.
    return seq_run(seeding_opts, finding_opts, propagation_opts, input_opts,
                   detector_opts, performance_opts, accelerator_opts);
}
