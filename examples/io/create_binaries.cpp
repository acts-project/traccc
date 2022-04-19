/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/reader.hpp"
#include "traccc/io/writer.hpp"
#include "traccc/options/handle_argument_errors.hpp"

namespace po = boost::program_options;

int create_binaries(const std::string& detector_file,
                    const std::string& cell_directory,
                    const std::string& hit_directory,
                    const unsigned int n_events, const int skip) {

    // Read the surface transforms
    auto surface_transforms = traccc::read_geometry(detector_file);

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Loop over events
    for (unsigned int event = skip; event < n_events + skip; ++event) {

        if (!cell_directory.empty()) {

            // Read the cells from the relevant event file
            traccc::host_cell_container cells_csv =
                traccc::read_cells_from_event(event, cell_directory,
                                              traccc::data_format::csv,
                                              surface_transforms, host_mr);

            // Write binary file
            traccc::write_cells(event, cell_directory,
                                traccc::data_format::binary, cells_csv);
        }

        if (!hit_directory.empty()) {

            // Read the hits from the relevant event file
            traccc::host_spacepoint_container spacepoints_csv =
                traccc::read_spacepoints_from_event(
                    event, hit_directory, traccc::data_format::csv,
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
    desc.add_options()("cell_directory",
                       po::value<std::string>()->default_value(""),
                       "specify the directory of cell files");
    desc.add_options()("hit_directory",
                       po::value<std::string>()->default_value(""),
                       "specify the directory of hit files");
    desc.add_options()("events", po::value<unsigned int>()->required(),
                       "number of events");
    desc.add_options()("skip", po::value<int>()->default_value(0),
                       "number of events to skip");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    auto detector_file = vm["detector_file"].as<std::string>();
    auto cell_directory = vm["cell_directory"].as<std::string>();
    auto hit_directory = vm["hit_directory"].as<std::string>();
    auto events = vm["events"].as<unsigned int>();
    auto skip = vm["skip"].as<int>();

    return create_binaries(detector_file, cell_directory, hit_directory, events,
                           skip);
}
