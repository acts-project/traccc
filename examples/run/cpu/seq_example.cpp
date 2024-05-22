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
#include "traccc/io/read_particles.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/write.hpp"

// algorithms
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation_algorithm.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// performance
#include "traccc/efficiency/seed_finding_efficiency.hpp"
#include "traccc/efficiency/track_finding_efficiency.hpp"
#include "traccc/performance/timer.hpp"

// options
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/output_data.hpp"
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
            const traccc::opts::output_data& output_opts,
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
                           detector_type::algebra_type,
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
    detray::propagation::config propagation_config(propagation_opts);

    finding_algorithm::config_type finding_cfg(finding_opts);
    finding_cfg.propagation = propagation_config;

    fitting_algorithm::config_type fitting_cfg;
    fitting_cfg.propagation = propagation_config;

    // Algorithms
    traccc::host::clusterization_algorithm ca(host_mr);
    traccc::host::spacepoint_formation_algorithm sf(host_mr);
    traccc::seeding_algorithm sa(seeding_opts.seedfinder,
                                 {seeding_opts.seedfinder},
                                 seeding_opts.seedfilter, host_mr);
    traccc::track_params_estimation tp(host_mr);
    finding_algorithm finding_alg(finding_cfg);
    fitting_algorithm fitting_alg(fitting_cfg);
    traccc::greedy_ambiguity_resolution_algorithm resolution_alg;

    // Performance analysis object(s).
    std::unique_ptr<traccc::performance::seed_finding_efficiency>
        seed_finding_efficiency;
    std::unique_ptr<traccc::performance::track_finding_efficiency>
        track_finding_efficiency;
    if (performance_opts.run) {
        seed_finding_efficiency =
            std::make_unique<traccc::performance::seed_finding_efficiency>(
                traccc::performance::seed_finding_efficiency::config{},
                traccc::performance::truth_filtering::config{},
                traccc::performance::truth_matching::config{});
        track_finding_efficiency =
            std::make_unique<traccc::performance::track_finding_efficiency>(
                traccc::performance::track_finding_efficiency::config{},
                traccc::performance::truth_filtering::config{},
                traccc::performance::truth_matching::config{});
    }

    // Timers
    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (unsigned int event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        traccc::host::clusterization_algorithm::output_type
            measurements_per_event{&host_mr};
        traccc::host::spacepoint_formation_algorithm::output_type
            spacepoints_per_event{&host_mr};
        traccc::seeding_algorithm::output_type seeds{&host_mr};
        traccc::track_params_estimation::output_type params{&host_mr};
        finding_algorithm::output_type track_candidates{&host_mr};
        fitting_algorithm::output_type track_states{&host_mr};
        traccc::greedy_ambiguity_resolution_algorithm::output_type
            resolved_track_states{&host_mr};

        {  // Start measuring wall time.
            traccc::performance::timer timer_wall{"Wall time", elapsedTimes};

            traccc::io::cell_reader_output readOut(&host_mr);

            {
                traccc::performance::timer timer{"Read cells", elapsedTimes};
                // Read the cells from the relevant event file
                traccc::io::read_cells(readOut, event, input_opts.directory,
                                       input_opts.format, &surface_transforms,
                                       &digi_cfg, barcode_map.get());
            }
            traccc::cell_collection_types::host& cells_per_event =
                readOut.cells;
            traccc::cell_module_collection_types::host& modules_per_event =
                readOut.modules;

            /*-------------------
                Clusterization
              -------------------*/

            {
                traccc::performance::timer timer{"Clusterization",
                                                 elapsedTimes};
                measurements_per_event =
                    ca(vecmem::get_data(cells_per_event),
                       vecmem::get_data(modules_per_event));
            }

            /*------------------------
                Spacepoint formation
              ------------------------*/

            {
                traccc::performance::timer timer{"Spacepoint formation",
                                                 elapsedTimes};
                spacepoints_per_event =
                    sf(vecmem::get_data(measurements_per_event),
                       vecmem::get_data(modules_per_event));
            }
            if (output_opts.directory != "") {
                traccc::io::write(event, output_opts.directory,
                                  output_opts.format,
                                  vecmem::get_data(spacepoints_per_event),
                                  vecmem::get_data(modules_per_event));
            }

            /*-----------------------
              Seeding algorithm
              -----------------------*/

            {
                traccc::performance::timer timer{"Seeding", elapsedTimes};
                seeds = sa(spacepoints_per_event);
            }
            if (output_opts.directory != "") {
                traccc::io::write(event, output_opts.directory,
                                  output_opts.format, vecmem::get_data(seeds),
                                  vecmem::get_data(spacepoints_per_event));
            }

            /*----------------------------
              Track params estimation
              ----------------------------*/

            {
                traccc::performance::timer timer{"Track params estimation",
                                                 elapsedTimes};
                params = tp(spacepoints_per_event, seeds, field_vec);
            }

            // Perform track finding and fitting only when using a Detray
            // geometry.
            if (detector_opts.use_detray_detector) {
                {
                    traccc::performance::timer timer{"Track finding",
                                                     elapsedTimes};
                    track_candidates = finding_alg(
                        detector, field, measurements_per_event, params);
                }
                if (output_opts.directory != "") {
                    traccc::io::write(
                        event, output_opts.directory, output_opts.format,
                        traccc::get_data(track_candidates), detector);
                }
                {
                    traccc::performance::timer timer{"Track fitting",
                                                     elapsedTimes};
                    track_states =
                        fitting_alg(detector, field, track_candidates);
                }
            }

            // Perform ambiguity resolution only if asked for.
            if (resolution_opts.run) {
                traccc::performance::timer timer{"Track ambiguity resolution",
                                                 elapsedTimes};
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

        }  // Stop measuring Wall time.

        /*--------------------
          Performance Analysis
          --------------------*/
        if (performance_opts.run) {

            // Read in the truth particles.
            traccc::particle_container_types::host truth_particles{&host_mr};
            traccc::io::read_particles(truth_particles, event,
                                       input_opts.directory, input_opts.format,
                                       barcode_map.get());

            // Analyze the reconstructed objects vs. the truth particles.
            seed_finding_efficiency->analyze(
                vecmem::get_data(seeds),
                vecmem::get_data(spacepoints_per_event),
                traccc::get_data(truth_particles));
            track_finding_efficiency->analyze(
                traccc::get_data(track_candidates),
                traccc::get_data(truth_particles));
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
    std::cout << "==> Elapsed times...\n" << elapsedTimes << std::endl;

    return EXIT_SUCCESS;
}

// The main routine
//
int main(int argc, char* argv[]) {

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::output_data output_opts{traccc::data_format::obj, ""};
    traccc::opts::clusterization clusterization_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::track_resolution resolution_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain on the Host",
        {detector_opts, input_opts, output_opts, clusterization_opts,
         seeding_opts, finding_opts, propagation_opts, resolution_opts,
         performance_opts},
        argc,
        argv};

    // Run the application.
    return seq_run(input_opts, output_opts, detector_opts, clusterization_opts,
                   seeding_opts, finding_opts, propagation_opts,
                   resolution_opts, performance_opts);
}
