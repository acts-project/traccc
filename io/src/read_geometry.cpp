/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_geometry.hpp"

#include "traccc/io/details/read_surfaces.hpp"
#include "traccc/io/utils.hpp"

namespace traccc::io {

geometry read_geometry(std::string_view filename, data_format format) {

    // Construct the full file name.
    const std::string full_filename = data_directory() + filename.data();

    // Read the file using another function. Relying on the auto-conversion of
    // the output type of that other function.
    return details::read_surfaces(full_filename, format);
}

}  // namespace traccc::io
