/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// io
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/io/utils.hpp"

// algorithms
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// performance
#include "traccc/efficiency/seeding_performance_writer.hpp"

// options
#include "traccc/options/common_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/seeding_input_options.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/toy_metadata.hpp"
#include "detray/io/common/detector_reader.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <iostream>

namespace po = boost::program_options;

int seq_run(const traccc::seeding_input_config& i_cfg,
            const traccc::common_options& common_opts) {
    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Read the surface transforms
    traccc::geometry surface_transforms;

    if (i_cfg.run_detray_geometry == false) {
        surface_transforms = traccc::io::read_geometry(i_cfg.detector_file);
    } else if (i_cfg.run_detray_geometry == true) {
        // Declare detector type
        using detector_t = detray::detector<detray::toy_metadata<>>;

        // Read the detector
        detray::io::detector_reader_config reader_cfg{};
        reader_cfg.add_file(traccc::io::data_directory() + i_cfg.detector_file)
            .add_file(traccc::io::data_directory() + i_cfg.material_file);

        const auto [det, names] =
            detray::io::read_detector<detector_t>(host_mr, reader_cfg);

        surface_transforms = traccc::io::alt_read_geometry(det);
    }

    // Output stats
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;

    // Configs
    traccc::seedfinder_config finder_config;
    traccc::spacepoint_grid_config grid_config(finder_config);
    traccc::seedfilter_config filter_config;

    traccc::seeding_algorithm sa(finder_config, grid_config, filter_config,
                                 host_mr);
    traccc::track_params_estimation tp(host_mr);

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});

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

        /*----------------
             Statistics
          ---------------*/

        n_spacepoints += spacepoints_per_event.size();
        n_seeds += seeds.size();

        /*------------
          Writer
          ------------*/

        if (common_opts.check_performance) {

            if (i_cfg.run_detray_geometry) {
                traccc::event_map2 evt_map(event, common_opts.input_directory,
                                           common_opts.input_directory,
                                           common_opts.input_directory);
                sd_performance_writer.write(
                    vecmem::get_data(seeds),
                    vecmem::get_data(spacepoints_per_event), evt_map);
            } else {
                traccc::event_map evt_map(event, i_cfg.detector_file,
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
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints" << std::endl;
    std::cout << "- created (cpu)  " << n_seeds << " seeds" << std::endl;

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

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    common_opts.read(vm);
    seeding_input_cfg.read(vm);

    std::cout << "Running " << argv[0] << " " << seeding_input_cfg.detector_file
              << " " << common_opts.input_directory << " " << common_opts.events
              << std::endl;

    return seq_run(seeding_input_cfg, common_opts);
}