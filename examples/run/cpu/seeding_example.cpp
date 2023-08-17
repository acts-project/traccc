/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
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
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// performance
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"

// options
#include "traccc/options/common_options.hpp"
#include "traccc/options/finding_input_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/propagation_options.hpp"
#include "traccc/options/seeding_input_options.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/toy_metadata.hpp"
#include "detray/io/common/detector_reader.hpp"
#include "detray/propagator/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <iostream>

namespace po = boost::program_options;

int seq_run(const traccc::seeding_input_config& /*i_cfg*/,
            const traccc::finding_input_config& finding_cfg,
            const traccc::propagation_options<traccc::scalar>& propagation_opts,
            const traccc::common_options& common_opts) {

    /// Type declarations
    using host_detector_type = detray::detector<detray::toy_metadata<>>;

    using b_field_t = typename host_detector_type::bfield_type;
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t,
                           typename host_detector_type::transform3,
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

    // Output stats
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_found_tracks = 0;
    uint64_t n_fitted_tracks = 0;

    /*****************************
     * Build a geometry
     *****************************/

    // B field value and its type
    // @TODO: Set B field as argument
    const traccc::vector3 B{0, 0, 2 * detray::unit<traccc::scalar>::T};

    // Declare detector
    host_detector_type host_det{host_mr};

    // Read the surface transforms
    traccc::geometry surface_transforms;

    if (not common_opts.run_detray_geometry) {
        surface_transforms =
            traccc::io::read_geometry(common_opts.detector_file);
    } else {
        // Read the detector
        detray::io::detector_reader_config reader_cfg{};
        reader_cfg
            .add_file(traccc::io::data_directory() + common_opts.detector_file)
            .add_file(traccc::io::data_directory() + common_opts.material_file)
            .bfield_vec(B[0], B[1], B[2]);

        auto [det, names] =
            detray::io::read_detector<host_detector_type>(host_mr, reader_cfg);
        host_det = std::move(det);

        surface_transforms = traccc::io::alt_read_geometry(host_det);
    }

    // Seeding algorithm
    traccc::seedfinder_config finder_config;
    traccc::spacepoint_grid_config grid_config(finder_config);
    traccc::seedfilter_config filter_config;

    traccc::seeding_algorithm sa(finder_config, grid_config, filter_config,
                                 host_mr);
    traccc::track_params_estimation tp(host_mr);

    // Finding algorithm configuration
    typename traccc::finding_algorithm<rk_stepper_type,
                                       host_navigator_type>::config_type cfg;

    cfg.min_track_candidates_per_track = finding_cfg.track_candidates_range[0];
    cfg.max_track_candidates_per_track = finding_cfg.track_candidates_range[1];
    cfg.constrained_step_size = propagation_opts.step_constraint;

    traccc::finding_algorithm<rk_stepper_type, host_navigator_type>
        host_finding(cfg);

    // Fitting algorithm object
    typename traccc::fitting_algorithm<host_fitter_type>::config_type fit_cfg;
    fit_cfg.step_constraint = propagation_opts.step_constraint;
    traccc::fitting_algorithm<host_fitter_type> host_fitting(fit_cfg);

    // Loop over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {

        // Read the hits from the relevant event file
        traccc::io::spacepoint_reader_output readOut(&host_mr);
        traccc::io::read_spacepoints(
            readOut, event, common_opts.input_directory, surface_transforms,
            common_opts.input_data_format);
        traccc::spacepoint_collection_types::host& spacepoints_per_event =
            readOut.spacepoints;

        /*----------------
             Seeding
          ---------------*/

        auto seeds = sa(spacepoints_per_event);

        /*----------------------------
           Track Parameter Estimation
          ----------------------------*/

        auto params = tp(spacepoints_per_event, seeds, readOut.modules,
                         {0.f, 0.f, finder_config.bFieldInZ});

        // Run CKF and KF if we are using a detray geometry
        traccc::track_candidate_container_types::host track_candidates;
        traccc::track_state_container_types::host track_states;

        if (common_opts.run_detray_geometry) {

            // Read measurements
            traccc::measurement_container_types::host measurements_per_event =
                traccc::io::read_measurements_container(
                    event, common_opts.input_directory,
                    traccc::data_format::csv, &host_mr);

            /*------------------------
               Track Finding with CKF
              ------------------------*/

            track_candidates =
                host_finding(host_det, measurements_per_event, params);
            n_found_tracks += track_candidates.size();

            /*------------------------
               Track Fitting with KF
              ------------------------*/

            track_states = host_fitting(host_det, track_candidates);
            n_fitted_tracks += track_states.size();
        }
        /*------------
           Statistics
          ------------*/

        n_spacepoints += spacepoints_per_event.size();
        n_seeds += seeds.size();

        /*------------
          Writer
          ------------*/

        if (common_opts.check_performance) {

            if (common_opts.run_detray_geometry) {

                traccc::event_map2 evt_map(event, common_opts.input_directory,
                                           common_opts.input_directory,
                                           common_opts.input_directory);
                sd_performance_writer.write(
                    vecmem::get_data(seeds),
                    vecmem::get_data(spacepoints_per_event), evt_map);

                find_performance_writer.write(
                    traccc::get_data(track_candidates), evt_map);

                for (unsigned int i = 0; i < track_states.size(); i++) {
                    const auto& trk_states_per_track = track_states.at(i).items;

                    const auto& fit_info = track_states[i].header;

                    fit_performance_writer.write(trk_states_per_track, fit_info,
                                                 host_det, evt_map);
                }
            } else {
                traccc::event_map evt_map(event, common_opts.detector_file,
                                          common_opts.input_directory,
                                          common_opts.input_directory, host_mr);
                sd_performance_writer.write(
                    vecmem::get_data(seeds),
                    vecmem::get_data(spacepoints_per_event), evt_map);
            }
        }
    }

    if (common_opts.check_performance) {
        sd_performance_writer.finalize();
        find_performance_writer.finalize();
        fit_performance_writer.finalize();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints" << std::endl;
    std::cout << "- created (cpu)  " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (cpu)  " << n_found_tracks << " found tracks"
              << std::endl;
    std::cout << "- created (cpu)  " << n_fitted_tracks << " fitted tracks"
              << std::endl;

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
    traccc::seeding_input_config seeding_input_cfg(desc);
    traccc::finding_input_config finding_input_cfg(desc);
    traccc::propagation_options<traccc::scalar> propagation_opts(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    common_opts.read(vm);
    seeding_input_cfg.read(vm);
    finding_input_cfg.read(vm);
    propagation_opts.read(vm);

    std::cout << "Running " << argv[0] << " " << common_opts.detector_file
              << " " << common_opts.input_directory << " " << common_opts.events
              << std::endl;

    return seq_run(seeding_input_cfg, finding_input_cfg, propagation_opts,
                   common_opts);
}