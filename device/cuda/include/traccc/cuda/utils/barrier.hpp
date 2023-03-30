/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

namespace traccc::cuda {

struct barrier {
    TRACCC_DEVICE
    void blockBarrier() { __syncthreads(); }

    TRACCC_DEVICE
    bool blockOr(bool predicate) { return __syncthreads_or(predicate); }
};

}  // namespace traccc::cuda