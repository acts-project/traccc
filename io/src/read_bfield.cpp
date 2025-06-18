/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_bfield.hpp"

// Covfie include(s).
#include <covfie/core/field.hpp>

namespace traccc::io {

/// @brief Function that reads the first 4 bytes of a potential bfield file and
/// checks that it contains data for a covfie field
bool check_covfie_file(const std::string& file_name) {

    // Open binary file
    detray::io::file_handle file{file_name,
                                 std::ios_base::in | std::ios_base::binary};

    // See "covfie/lib/core/utility/binary_io.hpp"
    std::uint32_t hdr = covfie::utility::read_binary<std::uint32_t>(*file);

    // Compare to magic bytes
    return (hdr == covfie::utility::MAGIC_HEADER);
}

}  // namespace traccc::io
