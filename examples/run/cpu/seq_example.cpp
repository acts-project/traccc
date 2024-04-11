/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// io
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"

// algorithms
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// performance
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"

// options
#include "traccc/options/clusterization.hpp"
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
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <memory>

int seq_run(const traccc::opts::input_data& input_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::clusterization& /*clusterization_opts*/,
            const traccc::opts::track_seeding& seeding_opts,
            const traccc::opts::track_finding& finding_opts,
            const traccc::opts::track_propagation& propagation_opts,
            const traccc::opts::track_resolution& resolution_opts,
            const traccc::opts::performance& performance_opts) {

    // Memory resource used by the application.
    vecmem::host_memory_resource host_mr;

    // Read in the geometry.
    auto [surface_transforms, barcode_map] = traccc::io::read_geometry(
        detector_opts.detector_file,
        (detector_opts.use_detray_detector ? traccc::data_format::json
                                           : traccc::data_format::csv));

    using detector_type = detray::detector<detray::default_metadata,
                                           detray::host_container_types>;
    detector_type detector{host_mr};
    if (detector_opts.use_detray_detector) {
        // Set up the detector reader configuration.
        detray::io::detector_reader_config cfg;
        cfg.add_file(traccc::io::data_directory() +
                     detector_opts.detector_file);
        if (detector_opts.material_file.empty() == false) {
            cfg.add_file(traccc::io::data_directory() +
                         detector_opts.material_file);
        }
        if (detector_opts.grid_file.empty() == false) {
            cfg.add_file(traccc::io::data_directory() +
                         detector_opts.grid_file);
        }

        // Read the detector.
        auto det = detray::io::read_detector<detector_type>(host_mr, cfg);
        detector = std::move(det.first);
    }

    // Read the digitization configuration file
    auto digi_cfg =
        traccc::io::read_digitization_config(detector_opts.digitization_file);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_found_tracks = 0;
    uint64_t n_fitted_tracks = 0;
    uint64_t n_ambiguity_free_tracks = 0;

    // Type definitions
    using stepper_type =
        detray::rk_stepper<detray::bfield::const_field_t::view_t,
                           detector_type::transform3,
                           detray::constrained_step<>>;
    using navigator_type = detray::navigator<const detector_type>;
    using finding_algorithm =
        traccc::finding_algorithm<stepper_type, navigator_type>;
    using fitting_algorithm = traccc::fitting_algorithm<
        traccc::kalman_fitter<stepper_type, navigator_type>>;

    // Constant B field for the track finding and fitting
    const traccc::vector3 field_vec = {0.f, 0.f,
                                       seeding_opts.seedfinder.bFieldInZ};
    const detray::bfield::const_field_t field =
        detray::bfield::create_const_field(field_vec);

    // Algorithm configuration(s).
    finding_algorithm::config_type finding_cfg;
    finding_cfg.min_track_candidates_per_track =
        finding_opts.track_candidates_range[0];
    finding_cfg.max_track_candidates_per_track =
        finding_opts.track_candidates_range[1];
    finding_cfg.chi2_max = finding_opts.chi2_max;
    finding_cfg.propagation = propagation_opts.config;

    fitting_algorithm::config_type fitting_cfg;
    fitting_cfg.propagation = propagation_opts.config;

    // Algorithms
    traccc::clusterization_algorithm ca(host_mr);
    traccc::spacepoint_formation sf(host_mr);
    traccc::seeding_algorithm sa(seeding_opts.seedfinder,
                                 {seeding_opts.seedfinder},
                                 seeding_opts.seedfilter, host_mr);
    traccc::track_params_estimation tp(host_mr);
    finding_algorithm finding_alg(finding_cfg);
    fitting_algorithm fitting_alg(fitting_cfg);
    traccc::greedy_ambiguity_resolution_algorithm resolution_alg;

    // performance writer
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

    // Loop over events
    for (unsigned int event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        traccc::io::cell_reader_output readOut(&host_mr);

        // Read the cells from the relevant event file
        traccc::io::read_cells(readOut, event, input_opts.directory,
                               input_opts.format, &surface_transforms,
                               &digi_cfg, barcode_map.get());
        traccc::cell_collection_types::host& cells_per_event = readOut.cells;
        traccc::cell_module_collection_types::host& modules_per_event =
            readOut.modules;

        /*-------------------
            Clusterization
          -------------------*/

        auto measurements_per_event = ca(cells_per_event, modules_per_event);

        /*------------------------
            Spacepoint formation
          ------------------------*/

        auto spacepoints_per_event =
            sf(measurements_per_event, modules_per_event);

        /*-----------------------
          Seeding algorithm
          -----------------------*/

        auto seeds = sa(spacepoints_per_event);

        /*----------------------------
          Track params estimation
          ----------------------------*/

        auto params = tp(spacepoints_per_event, seeds, field_vec);

        // Perform track finding and fitting only when using a Detray geometry.
        finding_algorithm::output_type track_candidates{&host_mr};
        fitting_algorithm::output_type track_states{&host_mr};
        if (detector_opts.use_detray_detector) {
            track_candidates =
                finding_alg(detector, field, measurements_per_event, params);
            track_states = fitting_alg(detector, field, track_candidates);
        }

        // Perform ambiguity resolution only if asked for.
        traccc::greedy_ambiguity_resolution_algorithm::output_type
            resolved_track_states{&host_mr};
        if (resolution_opts.run) {
            resolved_track_states = resolution_alg(track_states);
        }

        /*----------------------------
          Statistics
          ----------------------------*/

        n_modules += modules_per_event.size();
        n_cells += cells_per_event.size();
        n_measurements += measurements_per_event.size();
        n_spacepoints += spacepoints_per_event.size();
        n_seeds += seeds.size();
        n_found_tracks += track_candidates.size();
        n_fitted_tracks += track_states.size();
        n_ambiguity_free_tracks += resolved_track_states.size();

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

            for (unsigned int i = 0; i < track_states.size(); i++) {
                const auto& trk_states_per_track = track_states.at(i).items;

                const auto& fit_res = track_states[i].header;

                fit_performance_writer.write(trk_states_per_track, fit_res,
                                             detector, evt_map);
            }

            if (resolution_opts.run) {
                ar_performance_writer.write(
                    traccc::get_data(resolved_track_states), evt_map);
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
    std::cout << "- read     " << n_cells << " cells from " << n_modules
              << " modules" << std::endl;
    std::cout << "- created  " << n_measurements << " measurements. "
              << std::endl;
    std::cout << "- created  " << n_spacepoints << " space points. "
              << std::endl;
    std::cout << "- created  " << n_seeds << " seeds" << std::endl;
    std::cout << "- found    " << n_found_tracks << " tracks" << std::endl;
    std::cout << "- fitted   " << n_fitted_tracks << " tracks" << std::endl;
    std::cout << "- resolved " << n_ambiguity_free_tracks << " tracks"
              << std::endl;

    return EXIT_SUCCESS;
}

// The main routine
//
int main(int argc, char* argv[]) {

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::clusterization clusterization_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::track_resolution resolution_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain on the Host",
        {detector_opts, input_opts, clusterization_opts, seeding_opts,
         finding_opts, propagation_opts, resolution_opts, performance_opts},
        argc,
        argv};

    // Run the application.
    return seq_run(input_opts, detector_opts, clusterization_opts, seeding_opts,
                   finding_opts, propagation_opts, resolution_opts,
                   performance_opts);
}
