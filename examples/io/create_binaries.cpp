/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/reader.hpp"
#include "traccc/io/writer.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"

namespace po = boost::program_options;

int create_binaries(const std::string& detector_file,
                    const std::string& digi_config_file,
                    const std::string& cell_directory,
                    const std::string& hit_directory,
                    const traccc::common_options& common_opts) {

    // Read the surface transforms
    auto surface_transforms = traccc::read_geometry(detector_file);

    // Read the digitization configuration file
    auto digi_cfg = traccc::read_digitization_config(digi_config_file);

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Loop over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {

        if (!cell_directory.empty()) {

            // Read the cells from the relevant event file
            traccc::cell_container_types::host cells_csv =
                traccc::read_cells_from_event(
                    event, cell_directory, common_opts.input_data_format,
                    surface_transforms, digi_cfg, host_mr);

            // Write binary file
            traccc::write_cells(event, cell_directory,
                                traccc::data_format::binary, cells_csv);
        }

        if (!hit_directory.empty()) {

            // Read the hits from the relevant event file
            traccc::spacepoint_container_types::host spacepoints_csv =
                traccc::read_spacepoints_from_event(
                    event, hit_directory, common_opts.input_data_format,
                    surface_transforms, host_mr);

            // Write binary file
            traccc::write_spacepoints(event, hit_directory,
                                      traccc::data_format::binary,
                                      spacepoints_csv);
        }
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
    desc.add_options()("cell_directory",
                       po::value<std::string>()->default_value(""),
                       "specify the directory of cell files");
    desc.add_options()("hit_directory",
                       po::value<std::string>()->default_value(""),
                       "specify the directory of hit files");
    traccc::common_options common_opts(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    auto detector_file = vm["detector_file"].as<std::string>();
    auto digi_config_file = vm["digitization_config_file"].as<std::string>();
    auto cell_directory = vm["cell_directory"].as<std::string>();
    auto hit_directory = vm["hit_directory"].as<std::string>();
    common_opts.read(vm);

    return create_binaries(detector_file, digi_config_file, cell_directory,
                           hit_directory, common_opts);
}
