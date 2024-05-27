/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"

// io
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/io/utils.hpp"

// algorithms
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
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
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_resolution.hpp"
#include "traccc/options/track_seeding.hpp"

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
#include <iostream>

using namespace traccc;

int seq_run(const traccc::opts::track_seeding& seeding_opts,
            const traccc::opts::track_finding& finding_opts,
            const traccc::opts::track_propagation& propagation_opts,
            const traccc::opts::track_resolution& resolution_opts,
            const traccc::opts::input_data& input_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::performance& performance_opts) {

    /// Type declarations
    using host_detector_type = detray::detector<>;

    using b_field_t = covfie::field<detray::bfield::const_bknd_t>;
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t,
                           typename host_detector_type::algebra_type,
                           detray::constrained_step<>>;
    using host_navigator_type = detray::navigator<const host_detector_type>;
    using host_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, host_navigator_type>;

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
        detray::io::read_detector<host_detector_type>(host_mr, reader_cfg);

    traccc::geometry surface_transforms =
        traccc::io::alt_read_geometry(host_det);

    // Seeding algorithm
    traccc::seeding_algorithm sa(seeding_opts.seedfinder,
                                 {seeding_opts.seedfinder},
                                 seeding_opts.seedfilter, host_mr);
    traccc::track_params_estimation tp(host_mr);

    // Finding algorithm configuration
    typename traccc::finding_algorithm<rk_stepper_type,
                                       host_navigator_type>::config_type cfg;

    cfg.min_track_candidates_per_track = finding_opts.track_candidates_range[0];
    cfg.max_track_candidates_per_track = finding_opts.track_candidates_range[1];
    cfg.min_step_length_for_next_surface =
        finding_opts.min_step_length_for_next_surface;
    cfg.max_step_counts_for_next_surface =
        finding_opts.max_step_counts_for_next_surface;
    cfg.chi2_max = finding_opts.chi2_max;
    cfg.max_num_branches_per_seed = finding_opts.nmax_per_seed;
    cfg.max_num_skipping_per_cand = finding_opts.max_num_skipping_per_cand;
    cfg.propagation = propagation_opts.config;

    traccc::finding_algorithm<rk_stepper_type, host_navigator_type>
        host_finding(cfg);

    // Fitting algorithm object
    typename traccc::fitting_algorithm<host_fitter_type>::config_type fit_cfg;
    fit_cfg.propagation = propagation_opts.config;

    traccc::fitting_algorithm<host_fitter_type> host_fitting(fit_cfg);

    traccc::greedy_ambiguity_resolution_algorithm host_ambiguity_resolution{};

    // Loop over events
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Read the hits from the relevant event file
        traccc::io::spacepoint_reader_output readOut(&host_mr);
        traccc::io::read_spacepoints(readOut, event, input_opts.directory,
                                     surface_transforms, input_opts.format);
        traccc::spacepoint_collection_types::host& spacepoints_per_event =
            readOut.spacepoints;

        /*----------------
             Seeding
          ---------------*/

        auto seeds = sa(spacepoints_per_event);

        /*----------------------------
           Track Parameter Estimation
          ----------------------------*/

        auto params = tp(spacepoints_per_event, seeds,
                         {0.f, 0.f, seeding_opts.seedfinder.bFieldInZ});

        // Run CKF and KF if we are using a detray geometry
        traccc::track_candidate_container_types::host track_candidates;
        traccc::track_state_container_types::host track_states;
        traccc::track_state_container_types::host track_states_ar;

        // Read measurements
        traccc::io::measurement_reader_output meas_read_out(&host_mr);
        traccc::io::read_measurements(meas_read_out, event,
                                      input_opts.directory, input_opts.format);
        traccc::measurement_collection_types::host& measurements_per_event =
            meas_read_out.measurements;
        n_measurements += measurements_per_event.size();

        /*------------------------
           Track Finding with CKF
          ------------------------*/

        track_candidates =
            host_finding(host_det, field, measurements_per_event, params);
        n_found_tracks += track_candidates.size();

        /*------------------------
           Track Fitting with KF
          ------------------------*/

        track_states = host_fitting(host_det, field, track_candidates);
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

            traccc::event_map2 evt_map(event, input_opts.directory,
                                       input_opts.directory,
                                       input_opts.directory);
            sd_performance_writer.write(vecmem::get_data(seeds),
                                        vecmem::get_data(spacepoints_per_event),
                                        evt_map);

            find_performance_writer.write(traccc::get_data(track_candidates),
                                          evt_map);

            if (resolution_opts.run) {
                ar_performance_writer.write(traccc::get_data(track_states_ar),
                                            evt_map);
            }

            for (unsigned int i = 0; i < track_states.size(); i++) {
                const auto& trk_states_per_track = track_states.at(i).items;

                const auto& fit_res = track_states[i].header;

                fit_performance_writer.write(trk_states_per_track, fit_res,
                                             host_det, evt_map);
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

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints" << std::endl;
    std::cout << "- read    " << n_measurements << " measurements" << std::endl;
    std::cout << "- created (cpu)  " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (cpu)  " << n_found_tracks << " found tracks"
              << std::endl;
    std::cout << "- created (cpu)  " << n_fitted_tracks << " fitted tracks"
              << std::endl;

    if (resolution_opts.run) {
        std::cout << "- created (cpu)  " << n_ambiguity_free_tracks
                  << " ambiguity free tracks" << std::endl;
    } else {
        std::cout << "- ambiguity resolution: deactivated" << std::endl;
    }

    return EXIT_SUCCESS;
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
    traccc::opts::track_resolution resolution_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain on the Host (without clusterization)",
        {detector_opts, input_opts, seeding_opts, finding_opts,
         propagation_opts, resolution_opts, performance_opts},
        argc,
        argv};

    // Run the application.
    return seq_run(seeding_opts, finding_opts, propagation_opts,
                   resolution_opts, input_opts, detector_opts,
                   performance_opts);
}
