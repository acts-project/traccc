/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "write_track_parameters.hpp"

// System include(s).
#include <format>
#include <fstream>
#include <stdexcept>

namespace traccc::io::csv {

void write_track_parameters(
    std::string_view filename,
    bound_track_parameters_collection_types::const_view track_params_view) {

    // Open the file for writing.
    std::ofstream ofile(filename.data());
    if (!ofile.is_open()) {
        throw std::runtime_error(
            std::format("Could not open file \"{}\"", filename));
    }

    // Create device objects.
    const bound_track_parameters_collection_types::const_device track_params(
        track_params_view);

    // Write the header.
    ofile << "surface_link,local0,local1,phi,theta,time,qop\n";

    // Write out each track parameter.
    for (const traccc::bound_track_parameters& track : track_params) {

        // Write the track parameter info to the file.
        ofile << track.surface_link().value() << ","
              << track.bound_local().at(0) << "," << track.bound_local().at(1)
              << "," << track.phi() << "," << track.theta() << ","
              << track.time() << "," << track.qop() << '\n';
    }
}

}  // namespace traccc::io::csv
