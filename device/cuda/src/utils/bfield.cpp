/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/utils/bfield.hpp"

// Detray include(s).
#include "detray/io/utils/file_handle.hpp"

namespace traccc::cuda {

/// @brief Function that reads the first 4 bytes of a potential bfield file and
/// checks that it contains data for a covfie field
inline bool check_covfie_file(const std::string& file_name) {

    // Open binary file
    detray::io::file_handle file{file_name,
                                 std::ios_base::in | std::ios_base::binary};

    // See "covfie/lib/core/utility/binary_io.hpp"
    std::uint32_t hdr = covfie::utility::read_binary<std::uint32_t>(*file);

    // Compare to magic bytes
    return (hdr == covfie::utility::MAGIC_HEADER);
}

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

template <typename scalar_t>
inline inhom_bfield_backend_t<scalar_t> create_inhom_bfield() {
    return read_bfield<inhom_bfield_backend_t<scalar_t>>(
        !std::getenv("TRACCC_BFIELD_FILE") ? ""
                                           : std::getenv("TRACCC_BFIELD_FILE"));
}

template <typename scalar_t>
inline inhom_bfield_backend_t<scalar_t> create_inhom_bfield(
    const std::string& file_name) {
    return read_bfield<inhom_bfield_backend_t<scalar_t>>(file_name);
}

}  // namespace traccc::cuda