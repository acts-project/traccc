/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/details/configuration_printable.hpp"

// System include(s).
#include <sstream>

namespace traccc::opts {

configuration_printable::~configuration_printable() = default;

std::string configuration_printable::str() const {

    std::ostringstream out;
    print_impl(out, "", "", 0, get_max_key_width_impl());
    return out.str();
}

}  // namespace traccc::opts
