/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/performance/timing_info.hpp"

// System include(s).
#include <iomanip>
#include <iostream>

namespace traccc::performance {

/// Printout timer ids (string name) & elapsed time in miliseconds
std::ostream& operator<<(std::ostream& out, const timing_info& info) {
    for (auto i : info.data) {
        out << "\n"
            << std::setw(30) << std::right << i.first << "  " << std::setw(12)
            << std::left << (i.second.count() * 1.e-6);
    }
    return out;
}

}  // namespace traccc::performance
