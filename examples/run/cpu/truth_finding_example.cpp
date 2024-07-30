/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/utils/seed_generator.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/core/detector_metadata.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>

using namespace traccc;

int seq_run(const traccc::opts::track_finding& finding_opts,
            const traccc::opts::track_propagation& propagation_opts,
            const traccc::opts::input_data& input_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::performance& performance_opts) {

    /// Type declarations
    using host_detector_type = detray::detector<detray::default_metadata,
                                                detray::host_container_types>;

    using b_field_t = covfie::field<detray::bfield::const_bknd_t>;
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t, traccc::default_algebra,
                           detray::constrained_step<>>;

    using host_navigator_type = detray::navigator<const host_detector_type>;
    using host_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, host_navigator_type>;

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
    const auto [host_det, names] =
        detray::io::read_detector<host_detector_type>(host_mr, reader_cfg);

    const auto surface_transforms = traccc::io::alt_read_geometry(host_det);

    /*****************************
     * Do the reconstruction
     *****************************/

    // Standard deviations for seed track parameters
    static constexpr std::array<traccc::scalar, traccc::e_bound_size> stddevs =
        {1e-4f * detray::unit<traccc::scalar>::mm,
         1e-4f * detray::unit<traccc::scalar>::mm,
         1e-3f,
         1e-3f,
         1e-4f / detray::unit<traccc::scalar>::GeV,
         1e-4f * detray::unit<traccc::scalar>::ns};

    // Propagation configuration
    detray::propagation::config propagation_config(propagation_opts);

    // Finding algorithm configuration
    typename traccc::finding_algorithm<
        rk_stepper_type, host_navigator_type>::config_type cfg(finding_opts);
    cfg.propagation = propagation_config;

    // Finding algorithm object
    traccc::finding_algorithm<rk_stepper_type, host_navigator_type>
        host_finding(cfg);

    // Fitting algorithm object
    typename traccc::fitting_algorithm<host_fitter_type>::config_type fit_cfg;
    fit_cfg.propagation = propagation_config;

    traccc::fitting_algorithm<host_fitter_type> host_fitting(fit_cfg);

    // Seed generator
    traccc::seed_generator<host_detector_type> sg(host_det, stddevs);

    // Iterate over events
    for (unsigned int event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Truth Track Candidates
        traccc::event_map2 evt_map2(event, input_opts.directory,
                                    input_opts.directory, input_opts.directory);

        traccc::track_candidate_container_types::host truth_track_candidates =
            evt_map2.generate_truth_candidates(sg, host_mr);

        // Prepare truth seeds
        traccc::bound_track_parameters_collection_types::host seeds(&host_mr);
        const unsigned int n_tracks = truth_track_candidates.size();
        for (unsigned int i_trk = 0; i_trk < n_tracks; i_trk++) {
            seeds.push_back(truth_track_candidates.at(i_trk).header);
        }

        // Read measurements
        traccc::io::measurement_reader_output meas_read_out(&host_mr);
        traccc::io::read_measurements(meas_read_out, event,
                                      input_opts.directory, input_opts.format);
        traccc::measurement_collection_types::host& measurements_per_event =
            meas_read_out.measurements;

        // Run finding
        auto track_candidates =
            host_finding(host_det, field, measurements_per_event, seeds);

        std::cout << "Number of found tracks: " << track_candidates.size()
                  << std::endl;

        // Run fitting
        auto track_states = host_fitting(host_det, field, track_candidates);

        std::cout << "Number of fitted tracks: " << track_states.size()
                  << std::endl;

        const unsigned int n_fitted_tracks = track_states.size();

        if (performance_opts.run) {
            find_performance_writer.write(traccc::get_data(track_candidates),
                                          evt_map2);

            for (unsigned int i = 0; i < n_fitted_tracks; i++) {
                const auto& trk_states_per_track = track_states.at(i).items;

                const auto& fit_res = track_states[i].header;

                fit_performance_writer.write(trk_states_per_track, fit_res,
                                             host_det, evt_map2);
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

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::program_options program_opts{
        "Truth Track Finding on the Host",
        {detector_opts, input_opts, finding_opts, propagation_opts,
         performance_opts},
        argc,
        argv};

    // Run the application.
    return seq_run(finding_opts, propagation_opts, input_opts, detector_opts,
                   performance_opts);
}
