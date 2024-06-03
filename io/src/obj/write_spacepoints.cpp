/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "write_spacepoints.hpp"

// System include(s).
#include <fstream>

namespace traccc::io::obj {

void write_spacepoints(
    std::string_view filename,
    traccc::spacepoint_collection_types::const_view spacepoints_view) {

    // Open the output file.
    std::ofstream file(filename.data());
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " +
                                 std::string(filename));
    }

    // Create a device collection around the spacepoint view.
    traccc::spacepoint_collection_types::const_device spacepoints(
        spacepoints_view);

    // Write the spacepoints.
    for (const traccc::spacepoint& sp : spacepoints) {
        file << "v " << sp.x() << " " << sp.y() << " " << sp.z() << "\n";
    }
}

}  // namespace traccc::io::obj
