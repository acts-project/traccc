/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/utils/seed_generator.hpp"

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
#include <exception>
#include <iomanip>
#include <iostream>

using namespace traccc;

int seq_run(const traccc::opts::track_finding& finding_opts,
            const traccc::opts::track_propagation& propagation_opts,
            const traccc::opts::track_fitting& fitting_opts,
            const traccc::opts::input_data& input_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::performance& performance_opts,
            std::unique_ptr<const traccc::Logger> ilogger) {
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;

    // Performance writer
    traccc::finding_performance_writer find_performance_writer(
        traccc::finding_performance_writer::config{});
    traccc::fitting_performance_writer fit_performance_writer(
        traccc::fitting_performance_writer::config{});

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

    /*****************************
     * Do the reconstruction
     *****************************/

    // Standard deviations for seed track parameters
    static constexpr std::array<traccc::scalar, traccc::e_bound_size> stddevs =
        {1e-4f * traccc::unit<traccc::scalar>::mm,
         1e-4f * traccc::unit<traccc::scalar>::mm,
         1e-3f,
         1e-3f,
         1e-4f / traccc::unit<traccc::scalar>::GeV,
         1e-4f * traccc::unit<traccc::scalar>::ns};

    // Propagation configuration
    detray::propagation::config propagation_config(propagation_opts);

    // Finding algorithm configuration
    typename traccc::finding_config cfg(finding_opts);
    cfg.propagation = propagation_config;

    // Finding algorithm object
    traccc::host::combinatorial_kalman_filter_algorithm host_finding(
        cfg, logger().clone("FindingAlg"));

    // Fitting algorithm object
    traccc::fitting_config fit_cfg(fitting_opts);
    fit_cfg.propagation = propagation_config;

    traccc::host::kalman_fitting_algorithm host_fitting(
        fit_cfg, host_mr, logger().clone("FittingAlg"));

    // Seed generator
    traccc::seed_generator<traccc::default_detector::host> sg(detector,
                                                              stddevs);

    // Iterate over events
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Truth Track Candidates
        traccc::event_data evt_data(input_opts.directory, event, host_mr,
                                    input_opts.use_acts_geom_source, &detector,
                                    input_opts.format, false);

        traccc::track_candidate_container_types::host truth_track_candidates =
            evt_data.generate_truth_candidates(sg, host_mr);

        // Prepare truth seeds
        traccc::bound_track_parameters_collection_types::host seeds(&host_mr);
        const std::size_t n_tracks = truth_track_candidates.size();
        for (std::size_t i_trk = 0; i_trk < n_tracks; i_trk++) {
            seeds.push_back(
                truth_track_candidates.at(i_trk).header.seed_params);
        }

        // Read measurements
        traccc::measurement_collection_types::host measurements_per_event{
            &host_mr};
        traccc::io::read_measurements(
            measurements_per_event, event, input_opts.directory,
            (input_opts.use_acts_geom_source ? &detector : nullptr),
            input_opts.format);

        // Run finding
        auto track_candidates = host_finding(
            detector, field, vecmem::get_data(measurements_per_event),
            vecmem::get_data(seeds));

        std::cout << "Number of found tracks: " << track_candidates.size()
                  << std::endl;

        // Run fitting
        auto track_states =
            host_fitting(detector, field, traccc::get_data(track_candidates));

        std::cout << "Number of fitted tracks: ( "
                  << count_fitted_tracks(track_states) << " / "
                  << track_states.size() << " ) " << std::endl;

        const std::size_t n_fitted_tracks = track_states.size();

        if (performance_opts.run) {
            find_performance_writer.write(traccc::get_data(track_candidates),
                                          evt_data);

            for (std::size_t i = 0; i < n_fitted_tracks; i++) {
                const auto& trk_states_per_track = track_states.at(i).items;

                const auto& fit_res = track_states[i].header;

                fit_performance_writer.write(trk_states_per_track, fit_res,
                                             detector, evt_data);
            }
        }
    }

    if (performance_opts.run) {
        find_performance_writer.finalize();
        fit_performance_writer.finalize();
    }

    return EXIT_SUCCESS;
}

// The main routine
//
int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        "TracccExampleTruthFinding", traccc::Logging::Level::INFO);

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::track_fitting fitting_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::program_options program_opts{
        "Truth Track Finding on the Host",
        {detector_opts, input_opts, finding_opts, propagation_opts,
         fitting_opts, performance_opts},
        argc,
        argv,
        logger->cloneWithSuffix("Options")};

    // Run the application.
    return seq_run(finding_opts, propagation_opts, fitting_opts, input_opts,
                   detector_opts, performance_opts, logger->clone());
}
