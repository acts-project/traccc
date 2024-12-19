/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "write_track_candidates.hpp"

// System include(s).
#include <cassert>
#include <fstream>

namespace traccc::io::obj {

void write_track_candidates(
    std::string_view filename,
    track_candidate_container_types::const_view tracks_view,
    const traccc::default_detector::host& detector) {

    // Open the output file.
    std::ofstream file{filename.data()};
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " +
                                 std::string(filename));
    }

    // Create a device collection around the track container view.
    const track_candidate_container_types::const_device tracks{tracks_view};

    // Convenience type.
    using size_type = track_candidate_container_types::const_device::size_type;

    // First write out the measurements / spacepoints that the tracks are
    // made from. Don't try to resolve the overlaps, just write out duplicate
    // measurements if needed.
    file << "# Measurements / spacepoints that the tracks are made out of\n";
    for (size_type i = 0; i < tracks.size(); ++i) {

        // The track candidate in question.
        const track_candidate_container_types::const_device::const_element_view
            track = tracks.at(i);

        // Loop over the measurements that the track candidate is made out of.
        for (const measurement& m : track.items) {

            // Find the detector surface that this measurement sits on.
            const detray::tracking_surface surface{detector, m.surface_link};

            // Calculate a position for this measurement in global 3D space.
            const auto global = surface.bound_to_global({}, m.local, {});

            // Write the 3D coordinates of the measurement / spacepoint.
            assert(global.size() == 3);
            file << "v " << global[0] << " " << global[1] << " " << global[2]
                 << "\n";
        }
    }

    // Now loop over the track candidates again, and creates lines for each of
    // them using the measurements / spacepoints written out earlier.
    file << "# Track candidates\n";
    std::size_t vertex_counter = 1;
    for (size_type i = 0; i < tracks.size(); ++i) {

        // The track candidate in question.
        const track_candidate_container_types::const_device::const_element_view
            track = tracks.at(i);

        // Construct the lines.
        file << "l";
        for (size_type j = 0; j < track.items.size(); ++j) {
            file << " " << vertex_counter++;
        }
        file << "\n";
    }
}

}  // namespace traccc::io::obj
