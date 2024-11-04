/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "write_seeds.hpp"

// System include(s).
#include <fstream>

namespace traccc::io::obj {

void write_seeds(std::string_view filename,
                 seed_collection_types::const_view seeds_view,
                 spacepoint_collection_types::const_view spacepoints_view) {

    // Open the output file.
    std::ofstream file{filename.data()};
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " +
                                 std::string(filename));
    }

    // Create device collections around the views.
    const seed_collection_types::const_device seeds{seeds_view};
    const spacepoint_collection_types::const_device spacepoints{
        spacepoints_view};

    // Map associating in-memory spacepoint indices to in-file ones.
    std::map<std::size_t, std::size_t> spacepoint_indices;

    // Helper lambda to write a spacepoint to the output file.
    auto write_spacepoint =
        [&file, &spacepoints, &spacepoint_indices](
            spacepoint_collection_types::const_device::size_type memory_index,
            std::size_t file_index) -> bool {
        // Check whether this spacepoint has already been written.
        if (spacepoint_indices.find(memory_index) != spacepoint_indices.end()) {
            return false;
        }
        // Write the spacepoint.
        const traccc::spacepoint& sp = spacepoints[memory_index];
        file << "v " << sp.x() << " " << sp.y() << " " << sp.z() << "\n";
        // Remember the mapping.
        spacepoint_indices[memory_index] = file_index;
        return true;
    };

    // First, write out all of the spacepoints as vertices. Making sure that we
    // only write each spacepoint once. And remembering the indices of the
    // spacepoints in the output file, to be used when making the seeds.
    std::size_t file_index = 1;
    file << "# Spacepoints from which the seeds are built\n";
    for (const seed& s : seeds) {
        if (write_spacepoint(
                static_cast<
                    spacepoint_collection_types::const_device::size_type>(
                    s.spB_link),
                file_index)) {
            file_index++;
        }
        if (write_spacepoint(
                static_cast<
                    spacepoint_collection_types::const_device::size_type>(
                    s.spM_link),
                file_index)) {
            file_index++;
        }
        if (write_spacepoint(
                static_cast<
                    spacepoint_collection_types::const_device::size_type>(
                    s.spT_link),
                file_index)) {
            file_index++;
        }
    }

    // Helper lambda for getting an element of the spacepoint_indices map, in a
    // "safe" way.
    auto get_spacepoint_index =
        [&spacepoint_indices](std::size_t memory_index) -> std::size_t {
        auto it = spacepoint_indices.find(memory_index);
        if (it == spacepoint_indices.end()) {
            throw std::runtime_error("Spacepoint index not found");
        }
        return it->second;
    };

    // Now build the seeds as lines connecting the spacepoint vertices.
    file << "# Seeds\n";
    for (const seed& s : seeds) {
        file << "l " << get_spacepoint_index(s.spB_link) << " "
             << get_spacepoint_index(s.spM_link) << " "
             << get_spacepoint_index(s.spT_link) << "\n";
    }
}

}  // namespace traccc::io::obj
