/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/alpaka/finding/finding_algorithm.hpp"
#include "traccc/alpaka/fitting/fitting_algorithm.hpp"
#include "traccc/alpaka/seeding/seeding_algorithm.hpp"
#include "traccc/alpaka/seeding/track_params_estimation.hpp"
#include "traccc/alpaka/utils/get_vecmem_resource.hpp"
#include "traccc/definitions/common.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/efficiency/nseed_performance_writer.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/efficiency/track_filter.hpp"
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/accelerator.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/soa_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// Detray include(s).
#include <detray/detectors/bfield.hpp>
#include <detray/io/frontend/detector_reader.hpp>
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/propagator.hpp>
#include <detray/propagator/rk_stepper.hpp>

#ifdef ALPAKA_ACC_SYCL_ENABLED
#include <sycl/sycl.hpp>
#include <vecmem/utils/sycl/queue_wrapper.hpp>
#endif

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>

using namespace traccc;

int seq_run(const traccc::opts::track_seeding& seeding_opts,
            const traccc::opts::track_finding& finding_opts,
            const traccc::opts::track_propagation& propagation_opts,
            const traccc::opts::track_fitting& fitting_opts,
            const traccc::opts::input_data& input_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::performance& performance_opts,
            const traccc::opts::accelerator& accelerator_opts,
            [[maybe_unused]] std::unique_ptr<const traccc::Logger> ilogger) {
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    /// Type declarations
    using scalar_t = traccc::default_detector::host::scalar_type;
    using b_field_t = covfie::field<detray::bfield::const_bknd_t<scalar_t>>;
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t,
                           traccc::default_detector::host::algebra_type,
                           detray::constrained_step<scalar_t>>;
    using device_navigator_type =
        detray::navigator<const traccc::default_detector::device>;
    using device_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, device_navigator_type>;

#ifdef ALPAKA_ACC_SYCL_ENABLED
    ::sycl::queue q;
    vecmem::sycl::queue_wrapper qw{&q};
    traccc::alpaka::vecmem_resources::device_copy copy(qw);
    traccc::alpaka::vecmem_resources::host_memory_resource host_mr(qw);
    traccc::alpaka::vecmem_resources::device_memory_resource device_mr(qw);
    traccc::alpaka::vecmem_resources::managed_memory_resource mng_mr(qw);
    traccc::memory_resource mr{device_mr, &host_mr};
#else
    traccc::alpaka::vecmem_resources::device_copy copy;
    traccc::alpaka::vecmem_resources::host_memory_resource host_mr;
    traccc::alpaka::vecmem_resources::device_memory_resource device_mr;
    traccc::alpaka::vecmem_resources::managed_memory_resource mng_mr;
    traccc::memory_resource mr{device_mr, &host_mr};
#endif

    // Performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});
    traccc::finding_performance_writer find_performance_writer(
        traccc::finding_performance_writer::config{});
    traccc::fitting_performance_writer fit_performance_writer(
        traccc::fitting_performance_writer::config{});

    traccc::nseed_performance_writer nsd_performance_writer(
        "nseed_performance_",
        std::make_unique<traccc::simple_charged_eta_pt_cut>(
            2.7f, 1.f * traccc::unit<traccc::scalar>::GeV),
        std::make_unique<traccc::stepped_percentage>(0.6f));

    if (performance_opts.run) {
        nsd_performance_writer.initialize();
    }

    // Output stats
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_alpaka = 0;
    uint64_t n_found_tracks = 0;
    uint64_t n_found_tracks_alpaka = 0;
    uint64_t n_fitted_tracks = 0;
    uint64_t n_fitted_tracks_alpaka = 0;

    /*****************************
     * Build a geometry
     *****************************/

    // B field value and its type
    // @TODO: Set B field as argument
    const traccc::vector3 B{0, 0, 2 * traccc::unit<traccc::scalar>::T};
    auto field = detray::bfield::create_const_field<traccc::scalar>(B);

    // Construct a Detray detector object, if supported by the configuration.
    traccc::default_detector::host host_det{mng_mr};
    traccc::io::read_detector(host_det, mng_mr, detector_opts.detector_file,
                              detector_opts.material_file,
                              detector_opts.grid_file);

    // Detector view object
    traccc::default_detector::view det_view = detray::get_data(host_det);

    // Copy objects
    traccc::device::container_d2h_copy_alg<
        traccc::track_candidate_container_types>
        track_candidate_d2h{mr, copy,
                            logger().clone("TrackCandidateD2HCopyAlg")};

    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        track_state_d2h{mr, copy, logger().clone("TrackStateD2HCopyAlg")};

    // Seeding algorithms
    traccc::host::seeding_algorithm sa(
        seeding_opts.seedfinder, {seeding_opts.seedfinder},
        seeding_opts.seedfilter, host_mr, logger().clone("HostSeedingAlg"));
    traccc::host::track_params_estimation tp(
        host_mr, logger().clone("HostTrackParEstAlg"));

    // Alpaka Algorithms
    traccc::alpaka::seeding_algorithm sa_alpaka{
        seeding_opts.seedfinder,
        {seeding_opts.seedfinder},
        seeding_opts.seedfilter,
        mr,
        copy,
        logger().clone("AlpakaSeedingAlg")};
    traccc::alpaka::track_params_estimation tp_alpaka{
        mr, copy, logger().clone("AlpakaTrackParEstAlg")};

    // Propagation configuration
    detray::propagation::config propagation_config(propagation_opts);

    // Finding algorithm configuration
    traccc::finding_config cfg(finding_opts);
    cfg.propagation = propagation_config;

    // Finding algorithm object
    traccc::host::combinatorial_kalman_filter_algorithm host_finding(cfg);
    traccc::alpaka::finding_algorithm<rk_stepper_type, device_navigator_type>
        device_finding(cfg, mr, copy);

    // Fitting algorithm object
    traccc::fitting_config fit_cfg(fitting_opts);
    fit_cfg.propagation = propagation_config;

    traccc::host::kalman_fitting_algorithm host_fitting(fit_cfg, host_mr);
    traccc::alpaka::fitting_algorithm<device_fitter_type> device_fitting(
        fit_cfg, mr, copy);

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Instantiate host containers/collections
        traccc::edm::spacepoint_collection::host spacepoints_per_event{host_mr};
        traccc::measurement_collection_types::host measurements_per_event{
            &host_mr};
        traccc::host::seeding_algorithm::output_type seeds{host_mr};
        traccc::host::track_params_estimation::output_type params;
        traccc::track_candidate_container_types::host track_candidates;
        traccc::track_state_container_types::host track_states;

        traccc::edm::seed_collection::buffer seeds_alpaka_buffer;
        traccc::bound_track_parameters_collection_types::buffer
            params_alpaka_buffer(0, *mr.host);

        traccc::track_candidate_container_types::buffer
            track_candidates_alpaka_buffer{{{}, *(mr.host)},
                                           {{}, *(mr.host), mr.host}};

        traccc::track_state_container_types::buffer track_states_alpaka_buffer{
            {{}, *(mr.host)}, {{}, *(mr.host), mr.host}};

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
                    spacepoints_per_event, measurements_per_event, event,
                    input_opts.directory,
                    (input_opts.use_acts_geom_source ? &host_det : nullptr),
                    input_opts.format);

            }  // stop measuring hit reading timer

            /*----------------------------
                Seeding algorithm
            ----------------------------*/

            // Alpaka

            // Copy the spacepoint data to the device.
            traccc::edm::spacepoint_collection::buffer
                spacepoints_alpaka_buffer(
                    static_cast<unsigned int>(spacepoints_per_event.size()),
                    mr.main);
            copy(vecmem::get_data(spacepoints_per_event),
                 spacepoints_alpaka_buffer)
                ->wait();

            traccc::measurement_collection_types::buffer
                measurements_alpaka_buffer(
                    static_cast<unsigned int>(measurements_per_event.size()),
                    mr.main);
            copy(vecmem::get_data(measurements_per_event),
                 measurements_alpaka_buffer)
                ->wait();

            {
                traccc::performance::timer t("Seeding (alpaka)", elapsedTimes);
                // Reconstruct the spacepoints into seeds.
                seeds_alpaka_buffer =
                    sa_alpaka(vecmem::get_data(spacepoints_alpaka_buffer));
            }

            // CPU

            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Seeding  (cpu)", elapsedTimes);
                seeds = sa(vecmem::get_data(spacepoints_per_event));
            }  // stop measuring seeding cpu timer

            /*----------------------------
            Track params estimation
            ----------------------------*/

            // Alpaka

            {
                traccc::performance::timer t("Track params (alpaka)",
                                             elapsedTimes);
                params_alpaka_buffer =
                    tp_alpaka(measurements_alpaka_buffer,
                              spacepoints_alpaka_buffer, seeds_alpaka_buffer,
                              {0.f, 0.f, seeding_opts.seedfinder.bFieldInZ});
            }  // stop measuring track params alpaka timer

            // CPU
            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Track params  (cpu)",
                                             elapsedTimes);
                params = tp(vecmem::get_data(measurements_per_event),
                            vecmem::get_data(spacepoints_per_event),
                            vecmem::get_data(seeds),
                            {0.f, 0.f, seeding_opts.seedfinder.bFieldInZ});
            }  // stop measuring track params cpu timer

            /*------------------------
               Track Finding with CKF
              ------------------------*/

            {
                traccc::performance::timer t("Track finding with CKF (alpaka)",
                                             elapsedTimes);
                track_candidates_alpaka_buffer =
                    device_finding(det_view, field, measurements_alpaka_buffer,
                                   params_alpaka_buffer);
            }

            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Track finding with CKF (cpu)",
                                             elapsedTimes);
                track_candidates = host_finding(
                    host_det, field, vecmem::get_data(measurements_per_event),
                    vecmem::get_data(params));
            }

            /*------------------------
               Track Fitting with KF
              ------------------------*/

            {
                traccc::performance::timer t("Track fitting with KF (alpaka)",
                                             elapsedTimes);

                track_states_alpaka_buffer = device_fitting(
                    det_view, field, track_candidates_alpaka_buffer);
            }

            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Track fitting with KF (cpu)",
                                             elapsedTimes);
                track_states = host_fitting(host_det, field,
                                            traccc::get_data(track_candidates));
            }

        }  // Stop measuring wall time

        /*----------------------------------
          compare seeds from cpu and alpaka
          ----------------------------------*/

        // Copy the seeds to the host for comparisons
        traccc::edm::seed_collection::host seeds_alpaka{host_mr};
        traccc::bound_track_parameters_collection_types::host params_alpaka{
            &host_mr};
        copy(seeds_alpaka_buffer, seeds_alpaka)->wait();
        copy(params_alpaka_buffer, params_alpaka)->wait();

        // Copy track candidates from device to host
        traccc::track_candidate_container_types::host track_candidates_alpaka =
            track_candidate_d2h(track_candidates_alpaka_buffer);

        // Copy track states from device to host
        traccc::track_state_container_types::host track_states_alpaka =
            track_state_d2h(track_states_alpaka_buffer);

        if (accelerator_opts.compare_with_cpu) {
            // Show which event we are currently presenting the results for.
            std::cout << "===>>> Event " << event << " <<<===" << std::endl;

            // Compare the seeds made on the host and on the device
            traccc::soa_comparator<traccc::edm::seed_collection> compare_seeds{
                "seeds", traccc::details::comparator_factory<
                             traccc::edm::seed_collection::const_device::
                                 const_proxy_type>{
                             vecmem::get_data(spacepoints_per_event),
                             vecmem::get_data(spacepoints_per_event)}};
            compare_seeds(vecmem::get_data(seeds),
                          vecmem::get_data(seeds_alpaka));

            // Compare the track parameters made on the host and on the device.
            traccc::collection_comparator<traccc::bound_track_parameters<>>
                compare_track_parameters{"track parameters"};
            compare_track_parameters(vecmem::get_data(params),
                                     vecmem::get_data(params_alpaka));

            // Compare the track candidates made on the host and on the
            // device
            unsigned int n_matches = 0;
            for (unsigned int i = 0; i < track_candidates.size(); i++) {
                auto iso = traccc::details::is_same_object(
                    track_candidates.at(i).items);

                for (unsigned int j = 0; j < track_candidates_alpaka.size();
                     j++) {
                    if (iso(track_candidates_alpaka.at(j).items)) {
                        n_matches++;
                        break;
                    }
                }
            }
            std::cout << "Track candidate matching Rate: "
                      << float(n_matches) /
                             static_cast<float>(track_candidates.size())
                      << std::endl;
        }

        /*----------------
             Statistics
          ---------------*/

        n_spacepoints += spacepoints_per_event.size();
        n_seeds_alpaka += seeds_alpaka.size();
        n_seeds += seeds.size();
        n_found_tracks_alpaka += track_candidates_alpaka.size();
        n_found_tracks += track_candidates.size();
        n_fitted_tracks_alpaka += track_states_alpaka.size();
        n_fitted_tracks += track_states.size();

        /*------------
          Writer
          ------------*/

        if (performance_opts.run) {

            traccc::event_data evt_data(input_opts.directory, event, host_mr,
                                        input_opts.use_acts_geom_source,
                                        &host_det, input_opts.format, false);

            sd_performance_writer.write(
                vecmem::get_data(seeds),
                vecmem::get_data(spacepoints_per_event),
                vecmem::get_data(measurements_per_event), evt_data);
        }
    }

    if (performance_opts.run) {
        sd_performance_writer.finalize();
        nsd_performance_writer.finalize();
        find_performance_writer.finalize();
        fit_performance_writer.finalize();
        std::cout << nsd_performance_writer.generate_report_str();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints" << std::endl;
    std::cout << "- created  (cpu)  " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (alpaka)  " << n_seeds_alpaka << " seeds"
              << std::endl;
    std::cout << "- created  (cpu) " << n_found_tracks << " found tracks"
              << std::endl;
    std::cout << "- created (alpaka) " << n_found_tracks_alpaka
              << " found tracks" << std::endl;
    std::cout << "- created  (cpu) " << n_fitted_tracks << " fitted tracks"
              << std::endl;
    std::cout << "- created (alpaka) " << n_fitted_tracks_alpaka
              << " fitted tracks" << std::endl;
    std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        "TracccExampleSeedingAlpaka", traccc::Logging::Level::INFO);

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::track_fitting fitting_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::accelerator accelerator_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain Using Alpaka (without clusterization)",
        {detector_opts, input_opts, seeding_opts, finding_opts,
         propagation_opts, fitting_opts, performance_opts, accelerator_opts},
        argc,
        argv,
        logger->cloneWithSuffix("Options")};

    // Run the application.
    return seq_run(seeding_opts, finding_opts, propagation_opts, fitting_opts,
                   input_opts, detector_opts, performance_opts,
                   accelerator_opts, logger->clone());
}
