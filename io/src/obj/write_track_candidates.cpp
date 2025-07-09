/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "write_track_candidates.hpp"

// Detray include(s)
#include <detray/geometry/tracking_surface.hpp>

// System include(s).
#include <cassert>
#include <fstream>

namespace traccc::io::obj {

void write_track_candidates(
    std::string_view filename,
    edm::track_candidate_collection<default_algebra>::const_view tracks_view,
    measurement_collection_types::const_view measurements_view,
    const traccc::host_detector& detector) {

    // Open the output file.
    std::ofstream file{filename.data()};
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " +
                                 std::string(filename));
    }

    // Create a device collection around the track container view.
    const edm::track_candidate_collection<default_algebra>::const_device tracks{
        tracks_view};
    const measurement_collection_types::const_device measurements{
        measurements_view};

    // Convenience type.
    using size_type = edm::track_candidate_collection<
        default_algebra>::const_device::size_type;

    // First write out the measurements / spacepoints that the tracks are
    // made from. Don't try to resolve the overlaps, just write out duplicate
    // measurements if needed.
    file << "# Measurements / spacepoints that the tracks are made out of\n";
    for (size_type i = 0; i < tracks.size(); ++i) {

        // The track candidate in question.
        const edm::track_candidate_collection<
            default_algebra>::const_device::const_proxy_type track =
            tracks.at(i);

        // Loop over the measurements that the track candidate is made out of.
        for (unsigned int midx : track.measurement_indices()) {

            // The measurement in question.
            const measurement& m = measurements.at(midx);

            // Find the detector surface that this measurement sits on.
            const auto global = host_detector_visitor<detector_type_list>(
                detector, [&m]<typename detector_traits_t>(
                              const typename detector_traits_t::host& d) {
                    detray::tracking_surface surface{d, m.surface_link};
                    return surface.local_to_global({}, m.local, {});
                });

            // Write the 3D coordinates of the measurement / spacepoint.
            assert(global.size() == 3);
            file << "v " << global[0] << " " << global[1] << " " << global[2]
                 << "\n";
        }
    }

    // Now loop over the track candidates again, and creates lines for each
    // of them using the measurements / spacepoints written out earlier.
    file << "# Track candidates\n";
    std::size_t vertex_counter = 1;
    for (size_type i = 0; i < tracks.size(); ++i) {

        // The track candidate in question.
        const edm::track_candidate_collection<
            default_algebra>::const_device::const_proxy_type track =
            tracks.at(i);

        // Construct the lines.
        file << "l";
        for (size_type j = 0; j < track.measurement_indices().size(); ++j) {
            file << " " << vertex_counter++;
        }
        file << "\n";
    }
}

}  // namespace traccc::io::obj
