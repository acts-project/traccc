/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
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
namespace po = boost::program_options;

// The main routine
//
int main(int argc, char* argv[]) {

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::program_options program_opts{
        "Truth Track Fitting on the Host",
        {detector_opts, input_opts, propagation_opts, performance_opts},
        argc,
        argv};

    /// Type declarations
    using host_detector_type = detray::detector<detray::default_metadata,
                                                detray::host_container_types>;

    using b_field_t = covfie::field<detray::bfield::const_bknd_t>;
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t, traccc::transform3,
                           detray::constrained_step<>>;

    using host_navigator_type = detray::navigator<const host_detector_type>;
    using host_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, host_navigator_type>;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;

    // Performance writer
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

    /*****************************
     * Do the reconstruction
     *****************************/

    /// Standard deviations for seed track parameters
    static constexpr std::array<scalar, e_bound_size> stddevs = {
        0.03 * detray::unit<scalar>::mm,
        0.03 * detray::unit<scalar>::mm,
        0.017,
        0.017,
        0.01 / detray::unit<scalar>::GeV,
        1 * detray::unit<scalar>::ns};

    // Fitting algorithm object
    typename traccc::fitting_algorithm<host_fitter_type>::config_type fit_cfg;
    propagation_opts.setup(fit_cfg.propagation);

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

        // Run fitting
        auto track_states =
            host_fitting(host_det, field, truth_track_candidates);

        std::cout << "Number of fitted tracks: " << track_states.size()
                  << std::endl;

        const unsigned int n_fitted_tracks = track_states.size();

        if (performance_opts.run) {

            for (unsigned int i = 0; i < n_fitted_tracks; i++) {
                const auto& trk_states_per_track = track_states.at(i).items;

                const auto& fit_res = track_states[i].header;

                fit_performance_writer.write(trk_states_per_track, fit_res,
                                             host_det, evt_map2);
            }
        }
    }

    if (performance_opts.run) {
        fit_performance_writer.finalize();
    }

    return EXIT_SUCCESS;
}
