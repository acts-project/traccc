/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <iosfwd>

namespace traccc {

/// Format for an input or output file
enum data_format : int {
    csv = 0,
    binary = 1,
    json = 2,
};

/// Printout helper for @c traccc::data_format
std::ostream& operator<<(std::ostream& out, data_format format);

}  // namespace traccc
