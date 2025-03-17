/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/finding/details/debug_output.hpp"

#include "traccc/utils/logging.hpp"

namespace traccc {
void propagate_to_next_surface_debug::reset() {
    for (std::size_t i = 0;
         i < static_cast<std::size_t>(propagate_to_next_surface_exit_mode::MAX);
         ++i) {
        m_failure_count[i] = 0;
    }
}

void log_propagate_to_next_surface_debug(
    const propagate_to_next_surface_debug& obj, const Logger& logger) {
    if (unsigned int n = obj.m_failure_count[static_cast<std::size_t>(
            propagate_to_next_surface_exit_mode::HOLE_LIMIT_REACHED)];
        n > 0) {
        TRACCC_DEBUG(n << " tracks were killed due to reaching the hole limit");
    }

    if (unsigned int n = obj.m_failure_count[static_cast<std::size_t>(
            propagate_to_next_surface_exit_mode::BRANCH_LIMIT_REACHED)];
        n > 0) {
        TRACCC_DEBUG(
            n << " tracks were killed due to reaching the branch limit");
    }
}
}  // namespace traccc
