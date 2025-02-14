/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/geometry/detector.hpp"

// io
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/io/utils.hpp"

// algorithms
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// performance
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"

// options
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_resolution.hpp"
#include "traccc/options/track_seeding.hpp"

// Detray include(s).
#include <detray/core/detector.hpp>
#include <detray/detectors/bfield.hpp>
#include <detray/io/frontend/detector_reader.hpp>
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/propagator.hpp>
#include <detray/propagator/rk_stepper.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <cassert>
#include <cstdlib>
#include <iostream>

using namespace traccc;

int seq_run(const traccc::opts::track_seeding& seeding_opts,
            const traccc::opts::track_finding& finding_opts,
            const traccc::opts::track_propagation& propagation_opts,
            const traccc::opts::track_fitting& fitting_opts,
            const traccc::opts::track_resolution& resolution_opts,
            const traccc::opts::input_data& input_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::performance& performance_opts,
            std::unique_ptr<const traccc::Logger> ilogger) {
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});
    traccc::finding_performance_writer find_performance_writer(
        traccc::finding_performance_writer::config{});
    traccc::fitting_performance_writer fit_performance_writer(
        traccc::fitting_performance_writer::config{});

    traccc::finding_performance_writer::config ar_writer_cfg;
    ar_writer_cfg.file_path = "performance_track_ambiguity_resolution.root";
    ar_writer_cfg.algorithm_name = "ambiguity_resolution";
    traccc::finding_performance_writer ar_performance_writer(ar_writer_cfg);

    // Output stats
    uint64_t n_spacepoints = 0;
    uint64_t n_measurements = 0;
    uint64_t n_seeds = 0;
    uint64_t n_found_tracks = 0;
    uint64_t n_fitted_tracks = 0;
    uint64_t n_ambiguity_free_tracks = 0;

    /*****************************
     * Build a geometry
     *****************************/

    // B field value and its type
    // @TODO: Set B field as argument
    const traccc::vector3 B{0, 0, 2 * traccc::unit<traccc::scalar>::T};
    auto field = detray::bfield::create_const_field<traccc::scalar>(B);

    // Construct a Detray detector object, if supported by the configuration.
    traccc::default_detector::host detector{host_mr};
    assert(detector_opts.use_detray_detector == true);
    traccc::io::read_detector(detector, host_mr, detector_opts.detector_file,
                              detector_opts.material_file,
                              detector_opts.grid_file);

    // Seeding algorithm
    traccc::host::seeding_algorithm sa(
        seeding_opts.seedfinder, {seeding_opts.seedfinder},
        seeding_opts.seedfilter, host_mr, logger().clone("SeedingAlg"));
    traccc::host::track_params_estimation tp(host_mr,
                                             logger().clone("TrackParEstAlg"));

    // Propagation configuration
    detray::propagation::config propagation_config(propagation_opts);

    // Finding algorithm configuration
    traccc::finding_config cfg(finding_opts);
    cfg.propagation = propagation_config;

    traccc::host::combinatorial_kalman_filter_algorithm host_finding(
        cfg, logger().clone("FindingAlg"));

    // Fitting algorithm object
    traccc::fitting_config fit_cfg(fitting_opts);
    fit_cfg.propagation = propagation_config;

    traccc::host::kalman_fitting_algorithm host_fitting(
        fit_cfg, host_mr, logger().clone("FittingAlg"));

    traccc::greedy_ambiguity_resolution_algorithm::config_t
        host_ambiguity_config{};
    traccc::greedy_ambiguity_resolution_algorithm host_ambiguity_resolution(
        host_ambiguity_config, logger().clone("AmbiguityResolution"));

    // Loop over events
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Read the hits from the relevant event file
        traccc::measurement_collection_types::host measurements_per_event{
            &host_mr};
        traccc::edm::spacepoint_collection::host spacepoints_per_event{host_mr};
        traccc::io::read_spacepoints(
            spacepoints_per_event, measurements_per_event, event,
            input_opts.directory,
            (input_opts.use_acts_geom_source ? &detector : nullptr),
            input_opts.format);
        n_measurements += measurements_per_event.size();
        n_spacepoints += spacepoints_per_event.size();

        /*----------------
             Seeding
          ---------------*/

        auto seeds = sa(vecmem::get_data(spacepoints_per_event));

        /*----------------------------
           Track Parameter Estimation
          ----------------------------*/

        auto params =
            tp(vecmem::get_data(measurements_per_event),
               vecmem::get_data(spacepoints_per_event), vecmem::get_data(seeds),
               {0.f, 0.f, seeding_opts.seedfinder.bFieldInZ});

        // Run CKF and KF if we are using a detray geometry
        traccc::track_candidate_container_types::host track_candidates;
        traccc::track_state_container_types::host track_states;
        traccc::track_state_container_types::host track_states_ar;

        /*------------------------
           Track Finding with CKF
          ------------------------*/

        track_candidates = host_finding(
            detector, field, vecmem::get_data(measurements_per_event),
            vecmem::get_data(params));
        n_found_tracks += track_candidates.size();

        /*------------------------
           Track Fitting with KF
          ------------------------*/

        track_states =
            host_fitting(detector, field, traccc::get_data(track_candidates));
        n_fitted_tracks += track_states.size();

        /*-----------------------------------------
           Ambiguity Resolution with Greedy Solver
          -----------------------------------------*/

        if (resolution_opts.run) {
            track_states_ar = host_ambiguity_resolution(track_states);
            n_ambiguity_free_tracks += track_states_ar.size();
        }

        /*------------
           Statistics
          ------------*/

        n_spacepoints += spacepoints_per_event.size();
        n_seeds += seeds.size();

        /*------------
          Writer
          ------------*/

        if (performance_opts.run) {

            traccc::event_data evt_data(input_opts.directory, event, host_mr,
                                        input_opts.use_acts_geom_source,
                                        &detector, input_opts.format, false);

            sd_performance_writer.write(
                vecmem::get_data(seeds),
                vecmem::get_data(spacepoints_per_event),
                vecmem::get_data(measurements_per_event), evt_data);

            find_performance_writer.write(traccc::get_data(track_candidates),
                                          evt_data);

            if (resolution_opts.run) {
                ar_performance_writer.write(traccc::get_data(track_states_ar),
                                            evt_data);
            }

            for (unsigned int i = 0; i < track_states.size(); i++) {
                const auto& trk_states_per_track = track_states.at(i).items;

                const auto& fit_res = track_states[i].header;

                fit_performance_writer.write(trk_states_per_track, fit_res,
                                             detector, evt_data);
            }
        }
    }

    if (performance_opts.run) {
        sd_performance_writer.finalize();
        find_performance_writer.finalize();
        fit_performance_writer.finalize();
        if (resolution_opts.run) {
            ar_performance_writer.finalize();
        }
    }

    TRACCC_INFO("==> Statistics ... ");
    TRACCC_INFO("- read    " << n_spacepoints << " spacepoints");
    TRACCC_INFO("- read    " << n_measurements << " measurements");
    TRACCC_INFO("- created (cpu)  " << n_seeds << " seeds");
    TRACCC_INFO("- created (cpu)  " << n_found_tracks << " found tracks");
    TRACCC_INFO("- created (cpu)  " << n_fitted_tracks << " fitted tracks");

    if (resolution_opts.run) {
        TRACCC_INFO("- created (cpu)  " << n_ambiguity_free_tracks
                                        << " ambiguity free tracks");
    } else {
        TRACCC_INFO("- ambiguity resolution: deactivated");
    }

    return EXIT_SUCCESS;
}

// The main routine
//
int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        "TracccExampleSeeding", traccc::Logging::Level::INFO);

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::track_fitting fitting_opts;
    traccc::opts::track_resolution resolution_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain on the Host (without clusterization)",
        {detector_opts, input_opts, seeding_opts, finding_opts,
         propagation_opts, fitting_opts, resolution_opts, performance_opts},
        argc,
        argv,
        logger->cloneWithSuffix("Options")};

    // Run the application.
    return seq_run(seeding_opts, finding_opts, propagation_opts, fitting_opts,
                   resolution_opts, input_opts, detector_opts, performance_opts,
                   logger->clone());
}
