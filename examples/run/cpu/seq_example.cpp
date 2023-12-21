/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// io
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"

// algorithms
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// performance
#include "traccc/efficiency/seeding_performance_writer.hpp"

// options
#include "traccc/options/common_options.hpp"
#include "traccc/options/detector_input_options.hpp"
#include "traccc/options/finding_input_options.hpp"
#include "traccc/options/full_tracking_input_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/propagation_options.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/common/detector_reader.hpp"
#include "detray/propagator/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>

namespace po = boost::program_options;

int seq_run(const traccc::full_tracking_input_options& i_cfg,
            const traccc::common_options& common_opts,
            const traccc::detector_input_options& det_opts,
            const traccc::finding_input_options& finding_opts,
            const traccc::propagation_options& propagation_opts) {

    // Memory resource used by the application.
    vecmem::host_memory_resource host_mr;

    // Construct a detector object.
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(traccc::io::data_directory() + det_opts.detector_file);
    if (!det_opts.material_file.empty()) {
        reader_cfg.add_file(traccc::io::data_directory() +
                            det_opts.material_file);
    }
    if (!det_opts.grid_file.empty()) {
        reader_cfg.add_file(traccc::io::data_directory() + det_opts.grid_file);
    }
    auto [detector, _] =
        detray::io::read_detector<detray::detector<> >(host_mr, reader_cfg);

    // Construct an "old style geometry" from the detector object.
    traccc::geometry surface_transforms =
        traccc::io::alt_read_geometry(detector);

    // Construct a map from Acts surface identifiers to Detray barcodes.
    std::map<std::uint64_t, detray::geometry::barcode> barcode_map;
    for (const auto& surface : detector.surface_lookup()) {
        barcode_map[surface.source()] = surface.barcode();
    }

    // Read the digitization configuration file
    auto digi_cfg =
        traccc::io::read_digitization_config(i_cfg.digitization_config_file);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_found_tracks = 0;
    uint64_t n_fitted_tracks = 0;

    // Type definitions
    using stepper_type =
        detray::rk_stepper<detray::bfield::const_field_t::view_t,
                           detray::detector<>::transform3,
                           detray::constrained_step<>>;
    using navigator_type = detray::navigator<const detray::detector<>>;
    using finding_algorithm =
        traccc::finding_algorithm<stepper_type, navigator_type>;
    using fitting_algorithm = traccc::fitting_algorithm<
        traccc::kalman_fitter<stepper_type, navigator_type>>;

    // Constant B field for the track finding and fitting
    const traccc::vector3 field_vec = {0, 0,
                                       2 * detray::unit<traccc::scalar>::T};
    const detray::bfield::const_field_t field =
        detray::bfield::create_const_field(field_vec);

    // Configs
    traccc::seedfinder_config finder_config;
    traccc::spacepoint_grid_config grid_config(finder_config);
    traccc::seedfilter_config filter_config;

    finding_algorithm::config_type finding_config;
    finding_config.min_track_candidates_per_track =
        finding_opts.track_candidates_range[0];
    finding_config.max_track_candidates_per_track =
        finding_opts.track_candidates_range[1];
    finding_config.chi2_max = finding_opts.chi2_max;
    finding_config.step_constraint = propagation_opts.step_constraint;
    finding_config.overstep_tolerance = propagation_opts.overstep_tolerance;
    finding_config.mask_tolerance = propagation_opts.mask_tolerance;
    finding_config.rk_tolerance = propagation_opts.rk_tolerance;

    fitting_algorithm::config_type fitting_config;
    fitting_config.step_constraint = propagation_opts.step_constraint;
    fitting_config.overstep_tolerance = propagation_opts.overstep_tolerance;
    fitting_config.mask_tolerance = propagation_opts.mask_tolerance;
    fitting_config.rk_tolerance = propagation_opts.rk_tolerance;

    // Algorithms
    traccc::clusterization_algorithm ca(host_mr);
    traccc::spacepoint_formation sf(host_mr);
    traccc::seeding_algorithm sa(finder_config, grid_config, filter_config,
                                 host_mr);
    traccc::track_params_estimation tp(host_mr);
    finding_algorithm finding_alg(finding_config);
    fitting_algorithm fitting_alg(fitting_config);

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});

    // Loop over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {

        traccc::io::cell_reader_output readOut(&host_mr);

        // Read the cells from the relevant event file
        traccc::io::read_cells(readOut, event, common_opts.input_directory,
                               common_opts.input_data_format,
                               &surface_transforms, &digi_cfg, &barcode_map);
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

        /*-----------------------
          Track finding
          -----------------------*/

        auto track_candidates =
            finding_alg(detector, field, measurements_per_event, params);

        /*-----------------------
          Track fitting
          -----------------------*/

        auto track_states = fitting_alg(detector, field, track_candidates);

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

        /*------------
             Writer
          ------------*/

        if (common_opts.check_performance) {
            traccc::event_map evt_map(
                event, det_opts.detector_file, i_cfg.digitization_config_file,
                common_opts.input_directory, common_opts.input_directory,
                common_opts.input_directory, host_mr);

            sd_performance_writer.write(vecmem::get_data(seeds),
                                        vecmem::get_data(spacepoints_per_event),
                                        evt_map);
        }
    }

    if (common_opts.check_performance) {
        sd_performance_writer.finalize();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells from " << n_modules
              << " modules" << std::endl;
    std::cout << "- created " << n_measurements << " measurements. "
              << std::endl;
    std::cout << "- created " << n_spacepoints << " space points. "
              << std::endl;
    std::cout << "- created " << n_seeds << " seeds" << std::endl;
    std::cout << "- found   " << n_found_tracks << " tracks" << std::endl;
    std::cout << "- fitted  " << n_fitted_tracks << " tracks" << std::endl;

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
    traccc::detector_input_options det_opts(desc);
    traccc::full_tracking_input_options full_tracking_input_cfg(desc);
    traccc::finding_input_options finding_opts(desc);
    traccc::propagation_options propagation_opts(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    common_opts.read(vm);
    det_opts.read(vm);
    full_tracking_input_cfg.read(vm);
    finding_opts.read(vm);
    propagation_opts.read(vm);

    // Tell the user what's happening.
    std::cout << "\nRunning the full tracking chain on the host\n\n"
              << common_opts << "\n"
              << det_opts << "\n"
              << full_tracking_input_cfg << "\n"
              << finding_opts << "\n"
              << propagation_opts << "\n"
              << std::endl;

    return seq_run(full_tracking_input_cfg, common_opts, det_opts, finding_opts,
                   propagation_opts);
}
