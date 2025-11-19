/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "make_magnetic_field.hpp"

// Core include(s).
#include "traccc/geometry/detector_buffer.hpp"
#include "traccc/geometry/host_detector.hpp"
#include "traccc/utils/logging.hpp"

// Host algorithm(s).
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// Command line option include(s).
#include "traccc/options/accelerator.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/logging.hpp"
#include "traccc/options/magnetic_field.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/seed_matching.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_matching.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/options/truth_finding.hpp"

// Performance include(s).
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"

// I/O include(s).
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/read_spacepoints.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <cstdlib>

namespace traccc {

template <concepts::device_algorithm_maker ALG_MAKER>
int device_track_finding_validation(std::string_view logger_name,
                                    std::string_view description, int argc,
                                    char* argv[]) {

    // Logger object to use during the command line option reading.
    std::unique_ptr<const Logger> prelogger =
        getDefaultLogger(std::string{logger_name}, Logging::Level::INFO);

    // Program options.
    opts::detector detector_opts;
    opts::magnetic_field bfield_opts;
    opts::input_data input_opts;
    opts::track_seeding seeding_opts;
    opts::track_finding finding_opts;
    opts::track_propagation propagation_opts;
    opts::track_fitting fitting_opts;
    opts::performance performance_opts;
    opts::accelerator accelerator_opts;
    opts::truth_finding truth_finding_opts;
    opts::seed_matching seed_matching_opts;
    opts::track_matching track_matching_opts;
    opts::logging logging_opts;
    opts::program_options program_opts{
        description,
        {detector_opts, bfield_opts, input_opts, seeding_opts, finding_opts,
         propagation_opts, fitting_opts, performance_opts, accelerator_opts,
         truth_finding_opts, seed_matching_opts, track_matching_opts,
         logging_opts},
        argc,
        argv,
        prelogger->clone("options")};

    // The logger to use for the rest of the application.
    TRACCC_LOCAL_LOGGER(
        prelogger->clone(std::nullopt, Logging::Level(logging_opts)));

    // Create the algorithm maker.
    const ALG_MAKER alg_maker{logger().clone("algorithm_maker")};

    // Performance writer
    seeding_performance_writer seeding_pw(
        {.truth_config = truth_finding_opts,
         .seed_truth_config = seed_matching_opts},
        logger().clone("seeding_performance_writer"));
    finding_performance_writer finding_pw(
        {.truth_config = truth_finding_opts,
         .track_truth_config = track_matching_opts},
        logger().clone("finding_performance_writer"));
    finding_performance_writer postfit_finding_pw(
        {.file_path = "performance_track_postfit_finding.root",
         .truth_config = truth_finding_opts,
         .track_truth_config = track_matching_opts,
         .require_fit = true},
        logger().clone("post_fit_finding_performance_writer"));
    fitting_performance_writer fitting_pw(
        {}, logger().clone("fitting_performance_writer"));

    // Memory resource for the host algorithm(s).
    vecmem::host_memory_resource host_mr;
    // Copy object for the host algorithm(s).
    vecmem::copy host_copy;

    // Set up the detector description.
    silicon_detector_description::host det_descr{host_mr};
    io::read_detector_description(det_descr, detector_opts.detector_file,
                                  detector_opts.digitization_file,
                                  traccc::data_format::json);

    // Set up the magnetic field.
    const vector3 bfield_vec(seeding_opts);
    const auto host_field = details::make_magnetic_field(bfield_opts);
    const auto device_field = alg_maker.make_magnetic_field(
        host_field, accelerator_opts.use_gpu_texture_memory);

    // Set up the tracking geometry.
    host_detector host_det;
    io::read_detector(host_det, host_mr, detector_opts.detector_file,
                      detector_opts.material_file, detector_opts.grid_file);
    const detector_buffer device_det = buffer_from_host_detector(
        host_det, alg_maker.mr().main, alg_maker.copy());

    // Set up the seeding algorithm(s).
    const seedfinder_config sfinder_config(seeding_opts);
    const seedfilter_config sfilter_config(seeding_opts);
    const spacepoint_grid_config sg_config(seeding_opts);

    host::seeding_algorithm host_seeding(
        sfinder_config, sg_config, sfilter_config, host_mr,
        logger().clone("host::seeding_algorithm"));
    auto device_seeding = alg_maker.make_seeding_algorithm(
        sfinder_config, sg_config, sfilter_config);

    // Set up the track parameter estimation algorithm(s).
    const track_params_estimation_config tp_config;

    host::track_params_estimation host_tp_estimation(
        tp_config, host_mr, logger().clone("host::track_params_estimation"));
    auto device_tp_estimation =
        alg_maker.make_track_params_estimation_algorithm(tp_config);

    // Set up the track finding algorithm(s).
    const detray::propagation::config prop_config(propagation_opts);
    finding_config find_config(finding_opts);
    find_config.propagation = prop_config;

    host::combinatorial_kalman_filter_algorithm host_finding(
        find_config, host_mr,
        logger().clone("host::combinatorial_kalman_filter_algorithm"));
    auto device_finding = alg_maker.make_finding_algorithm(find_config);

    // Set up the track fitting algorithm(s).
    fitting_config fit_config(fitting_opts);
    fit_config.propagation = prop_config;

    host::kalman_fitting_algorithm host_fitting(
        fit_config, host_mr, host_copy,
        logger().clone("host::kalman_fitting_algorithm"));
    auto device_fitting = alg_maker.make_fitting_algorithm(fit_config);

    // Times elapsed in the various reconstruction steps.
    performance::timing_info times;

    // Process the requested number of events.
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Instantiate host containers/collections.
        edm::spacepoint_collection::host host_spacepoints{host_mr};
        edm::measurement_collection<default_algebra>::host host_measurements{
            host_mr};
        edm::seed_collection::host host_seeds{host_mr};
        bound_track_parameters_collection_types::host host_track_params{
            &host_mr};
        edm::track_container<default_algebra>::host host_found_tracks{host_mr};
        edm::track_container<default_algebra>::host host_fitted_tracks{host_mr};

        edm::spacepoint_collection::buffer device_spacepoints;
        edm::measurement_collection<default_algebra>::buffer
            device_measurements;
        edm::seed_collection::buffer device_seeds;
        bound_track_parameters_collection_types::buffer device_track_params;
        edm::track_container<default_algebra>::buffer device_found_tracks;
        edm::track_container<default_algebra>::buffer device_fitted_tracks;

        {
            // Measure the total wall time.
            performance::timer wall_t("Wall time", times);

            {
                // Read the spacepoints and measurements from the relevant event
                // files.
                performance::timer t("Host data reading", times);
                io::read_spacepoints(
                    host_spacepoints, host_measurements, event,
                    input_opts.directory,
                    (input_opts.use_acts_geom_source ? &host_det : nullptr),
                    &det_descr, input_opts.format);
            }

            {
                // Copy the spacepoint and measurement data to the device.
                performance::timer t{"Host->Device data transfers", times};
                device_spacepoints = alg_maker.copy().to(
                    vecmem::get_data(host_spacepoints), alg_maker.mr().main,
                    alg_maker.mr().host, vecmem::copy::type::host_to_device);
                device_measurements = alg_maker.copy().to(
                    vecmem::get_data(host_measurements), alg_maker.mr().main,
                    alg_maker.mr().host, vecmem::copy::type::host_to_device);
            }

            {
                // Reconstruct the spacepoints into seeds on the device.
                performance::timer t("Device seeding", times);
                device_seeds = (*device_seeding)(device_spacepoints);
                alg_maker.synchronize();
            }

            if (accelerator_opts.compare_with_cpu) {
                // Reconstruct the spacepoints into seeds on the host.
                performance::timer t("Host seeding", times);
                host_seeds = host_seeding(vecmem::get_data(host_spacepoints));
            }

            {
                // Run track parameter estimation on the device.
                performance::timer t("Device track parameter estimation",
                                     times);
                device_track_params = (*device_tp_estimation)(
                    device_measurements, device_spacepoints, device_seeds,
                    bfield_vec);
                alg_maker.synchronize();
            }

            if (accelerator_opts.compare_with_cpu) {
                // Run track parameter estimation on the host.
                performance::timer t("Host track parameter esimation", times);
                host_track_params = host_tp_estimation(
                    vecmem::get_data(host_measurements),
                    vecmem::get_data(host_spacepoints),
                    vecmem::get_data(host_seeds), bfield_vec);
            }

            {
                // Run track finding on the device.
                performance::timer t("Device track finding", times);
                device_found_tracks =
                    (*device_finding)(device_det, device_field,
                                      device_measurements, device_track_params);
            }

            if (accelerator_opts.compare_with_cpu) {
                // Run track finding on the host.
                traccc::performance::timer t("Host track finding", times);
                host_found_tracks = host_finding(
                    host_det, host_field, vecmem::get_data(host_measurements),
                    vecmem::get_data(host_track_params));
            }

            /*------------------------
               Track Fitting with KF
              ------------------------*/

            {
                traccc::performance::timer t("Track fitting with KF (cuda)",
                                             times);

                track_states_cuda_buffer = device_fitting(
                    device_det, device_field, track_candidates_cuda_buffer);
            }

            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Track fitting with KF (cpu)",
                                             times);
                track_states = host_fitting(
                    host_det, host_field,
                    traccc::edm::track_container<
                        traccc::default_algebra>::const_data(track_candidates));
            }

        }  // Stop measuring wall time

        /*----------------------------------
          compare seeds from cpu and cuda
          ----------------------------------*/

        // Copy the seeds to the host for comparisons
        traccc::edm::seed_collection::host seeds_cuda{host_mr};
        traccc::bound_track_parameters_collection_types::host params_cuda;
        async_copy(seeds_cuda_buffer, seeds_cuda)->wait();
        async_copy(params_cuda_buffer, params_cuda)->wait();

        // Copy track candidates from device to host
        traccc::edm::track_container<traccc::default_algebra>::host
            track_candidates_cuda{host_mr,
                                  vecmem::get_data(measurements_per_event)};
        async_copy(track_candidates_cuda_buffer.tracks,
                   track_candidates_cuda.tracks)
            ->wait();

        // Copy track states from device to host
        traccc::edm::track_container<traccc::default_algebra>::host
            track_states_cuda{host_mr};
        async_copy(track_states_cuda_buffer.tracks, track_states_cuda.tracks)
            ->wait();
        async_copy(track_states_cuda_buffer.states, track_states_cuda.states)
            ->wait();
        track_states_cuda.measurements =
            vecmem::get_data(measurements_per_event);

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
                          vecmem::get_data(seeds_cuda));

            // Compare the track parameters made on the host and on the device.
            traccc::collection_comparator<traccc::bound_track_parameters<>>
                compare_track_parameters{"track parameters"};
            compare_track_parameters(vecmem::get_data(params),
                                     vecmem::get_data(params_cuda));

            // Compare the track candidates made on the host and on the
            // device
            traccc::soa_comparator<
                traccc::edm::track_collection<traccc::default_algebra>>
                compare_track_candidates{
                    "track candidates",
                    traccc::details::comparator_factory<
                        traccc::edm::track_collection<traccc::default_algebra>::
                            const_device::const_proxy_type>{
                        vecmem::get_data(measurements_per_event),
                        vecmem::get_data(measurements_per_event),
                        {},
                        {}}};
            compare_track_candidates(
                vecmem::get_data(track_candidates.tracks),
                vecmem::get_data(track_candidates_cuda.tracks));
        }

        /*----------------
             Statistics
          ---------------*/

        n_spacepoints += spacepoints_per_event.size();
        n_seeds_cuda += seeds_cuda.size();
        n_seeds += seeds.size();
        n_found_tracks_cuda += track_candidates_cuda.tracks.size();
        n_found_tracks += track_candidates.tracks.size();
        n_fitted_tracks_cuda += track_states_cuda.tracks.size();
        n_fitted_tracks += track_states.tracks.size();

        /*------------
          Writer
          ------------*/

        if (performance_opts.run) {

            traccc::event_data evt_data(input_opts.directory, event, host_mr,
                                        input_opts.use_acts_geom_source,
                                        &host_det, input_opts.format, false);

            sd_performance_writer.write(
                vecmem::get_data(seeds_cuda),
                vecmem::get_data(spacepoints_per_event),
                vecmem::get_data(measurements_per_event), evt_data);

            find_performance_writer.write(
                traccc::edm::track_container<
                    traccc::default_algebra>::const_data(track_candidates_cuda),
                evt_data);

            postfit_find_performance_writer.write(
                traccc::edm::track_container<
                    traccc::default_algebra>::const_data(track_states_cuda),
                evt_data);

            for (unsigned int i = 0; i < track_states_cuda.tracks.size(); i++) {
                host_detector_visitor<detector_type_list>(
                    host_det, [&]<typename detector_traits_t>(
                                  const typename detector_traits_t::host& det) {
                        fit_performance_writer.write(
                            track_states_cuda.tracks.at(i),
                            track_states_cuda.states, measurements_per_event,
                            det, evt_data);
                    });
            }
        }
    }

    // Return gracefully.
    return EXIT_SUCCESS;
}

}  // namespace traccc
