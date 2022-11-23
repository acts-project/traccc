/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/performance/timing_info.hpp"

// System include(s).
#include <algorithm>
#include <exception>
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

/// Return time taken by timer identified with given name, in miliseconds
std::chrono::nanoseconds timing_info::get_time(
    const std::string_view timer_name) {
    const std::string name(timer_name);
    auto it =
        std::find_if(data.begin(), data.end(),
                     [&name](timing_info_pair it) { return it.first == name; });
    if (it == data.end()) {
        throw std::invalid_argument(
            "Called timing_info::get_timer with a name not listed: " + name);
    }
    return it->second;
}

}  // namespace traccc::performance
