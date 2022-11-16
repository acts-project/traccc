/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/details/read_surfaces.hpp"

#include "../csv/read_surfaces.hpp"

namespace traccc::io::details {

std::map<geometry_id, transform3> read_surfaces(std::string_view filename,
                                                data_format format) {

    // Decide how to read the file.
    switch (format) {
        case data_format::csv:
            return csv::read_surfaces(filename);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io::details
