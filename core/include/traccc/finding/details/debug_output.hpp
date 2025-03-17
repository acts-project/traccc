/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/utils/logging.hpp"

namespace traccc {
enum class propagate_to_next_surface_exit_mode {
    HOLE_LIMIT_REACHED,
    BRANCH_LIMIT_REACHED,
    MAX
};

struct propagate_to_next_surface_debug {
    void reset();

    unsigned int m_failure_count[static_cast<std::size_t>(
        propagate_to_next_surface_exit_mode::MAX)];
};

void log_propagate_to_next_surface_debug(
    const propagate_to_next_surface_debug& obj, const Logger& logger);
}  // namespace traccc
