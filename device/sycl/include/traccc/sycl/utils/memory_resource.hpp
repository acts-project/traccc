/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::sycl {

// Simple struct for different sycl memory resources
struct memory_resource {

    vecmem::memory_resource* host = nullptr;
    vecmem::memory_resource* device = nullptr;
};

}  // namespace traccc::sycl