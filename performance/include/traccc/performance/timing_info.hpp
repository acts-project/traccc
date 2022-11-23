/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <chrono>
#include <string_view>
#include <utility>
#include <vector>

namespace traccc::performance {

using timing_info_pair = std::pair<std::string, std::chrono::nanoseconds>;

/// Struct for storing time measurements collected in timer class
///
struct timing_info {
    std::vector<timing_info_pair> data;

    std::chrono::nanoseconds get_time(const std::string_view timer_name);
};

std::ostream& operator<<(std::ostream& out, const timing_info& info);

}  // namespace traccc::performance
