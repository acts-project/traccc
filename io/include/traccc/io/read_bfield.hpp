/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray include(s).
#include <detray/io/utils/file_handle.hpp>

// System include(s).
#include <iostream>

namespace traccc::io {

/// @brief Function that reads the first 4 bytes of a potential bfield file and
/// checks that it contains data for a covfie field
bool check_covfie_file(const std::string& file_name);

/// @brief function that reads a covfie field from file
template <typename bfield_t>
inline bfield_t read_bfield(const std::string& file_name) {

    if (!check_covfie_file(file_name)) {
        throw std::runtime_error("Not a valid covfie file: " + file_name);
    }

    // Open binary file
    detray::io::file_handle file{file_name,
                                 std::ios_base::in | std::ios_base::binary};

    return bfield_t(*file);
}

}  // namespace traccc::io
