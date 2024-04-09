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
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// performance
#include "traccc/efficiency/seeding_performance_writer.hpp"

// options
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"

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
            const traccc::opts::performance& performance_opts) {

    // Read in the geometry.
    auto [surface_transforms, barcode_map] = traccc::io::read_geometry(
        detector_opts.detector_file,
        (detector_opts.use_detray_detector ? traccc::data_format::json
                                           : traccc::data_format::csv));

    // Read the digitization configuration file
    auto digi_cfg =
        traccc::io::read_digitization_config(detector_opts.digitization_file);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;

    // Configs
    traccc::seedfinder_config finder_config;
    traccc::spacepoint_grid_config grid_config(finder_config);
    traccc::seedfilter_config filter_config;

    // Constant B field for the track finding and fitting
    const traccc::vector3 field_vec = {0.f, 0.f, finder_config.bFieldInZ};

    // Memory resource used by the application.
    vecmem::host_memory_resource host_mr;

    // Algorithms
    traccc::clusterization_algorithm ca(host_mr);
    traccc::spacepoint_formation sf(host_mr);
    traccc::seeding_algorithm sa(finder_config, grid_config, filter_config,
                                 host_mr);
    traccc::track_params_estimation tp(host_mr);

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});

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

        /*----------------------------
          Statistics
          ----------------------------*/

        n_modules += modules_per_event.size();
        n_cells += cells_per_event.size();
        n_measurements += measurements_per_event.size();
        n_spacepoints += spacepoints_per_event.size();
        n_seeds += seeds.size();

        /*------------
             Writer
          ------------*/

        if (performance_opts.run) {
            traccc::event_map evt_map(
                event, detector_opts.detector_file,
                detector_opts.digitization_file, input_opts.directory,
                input_opts.directory, input_opts.directory, host_mr);

            sd_performance_writer.write(vecmem::get_data(seeds),
                                        vecmem::get_data(spacepoints_per_event),
                                        evt_map);
        }
    }

    if (performance_opts.run) {
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

    return EXIT_SUCCESS;
}

// The main routine
//
int main(int argc, char* argv[]) {

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain on the Host",
        {detector_opts, input_opts, performance_opts},
        argc,
        argv};

    // Run the application.
    return seq_run(input_opts, detector_opts, performance_opts);
}
