/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstdint>

namespace traccc::device::details {
struct ccl_debug_output {
    uint32_t num_oversized_partitions;

    static ccl_debug_output init() {
        ccl_debug_output rv;

        rv.num_oversized_partitions = 0;

        return rv;
    }
};
}  // namespace traccc::device::details
