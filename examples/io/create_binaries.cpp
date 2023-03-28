/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints_alt.hpp"
#include "traccc/io/write.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

namespace po = boost::program_options;

int create_binaries(const std::string& detector_file,
                    const std::string& digi_config_file,
                    const traccc::common_options& common_opts) {

    // Read the surface transforms
    auto surface_transforms = traccc::io::read_geometry(detector_file);

    // Read the digitization configuration file
    auto digi_cfg = traccc::io::read_digitization_config(digi_config_file);

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Loop over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {

        // Read the cells from the relevant event file
        auto cells_csv = traccc::io::read_cells(
            event, common_opts.input_directory, common_opts.input_data_format,
            &surface_transforms, &digi_cfg, &host_mr);

        // Write binary file
        traccc::io::write(event, common_opts.input_directory,
                          traccc::data_format::binary,
                          vecmem::get_data(cells_csv.cells),
                          vecmem::get_data(cells_csv.modules));

        // Read the hits from the relevant event file
        auto spacepoints_csv = traccc::io::read_spacepoints_alt(
            event, common_opts.input_directory, surface_transforms,
            common_opts.input_data_format, &host_mr);

        // Write binary file
        traccc::io::write(event, common_opts.input_directory,
                          traccc::data_format::binary,
                          vecmem::get_data(spacepoints_csv.spacepoints),
                          vecmem::get_data(spacepoints_csv.modules));

        // Read the measurements from the relevant event file
        auto measurements_csv = traccc::io::read_measurements(
            event, common_opts.input_directory, common_opts.input_data_format,
            &host_mr);

        // Write binary file
        traccc::io::write(event, common_opts.input_directory,
                          traccc::data_format::binary,
                          vecmem::get_data(measurements_csv.measurements),
                          vecmem::get_data(measurements_csv.modules));
    }

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    desc.add_options()("detector_file", po::value<std::string>()->required(),
                       "specify detector file");
    desc.add_options()("digitization_config_file",
                       po::value<std::string>()->required(),
                       "specify digitization configuration file");
    traccc::common_options common_opts(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    auto detector_file = vm["detector_file"].as<std::string>();
    auto digi_config_file = vm["digitization_config_file"].as<std::string>();
    common_opts.read(vm);

    return create_binaries(detector_file, digi_config_file, common_opts);
}
