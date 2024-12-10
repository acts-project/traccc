/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/details/is_aligned.hpp"

// System include(s).
#include <memory>

namespace vecmem {
namespace details {

VECMEM_HOST
bool is_aligned(void* ptr, std::size_t alignment) {

    // Use std::align to see if the pointer needs to be modified to be aligned
    // to the requested value. Just use a dummy "size" and "space" parameter
    // for the call.
    const std::size_t size = 2 * alignment;
    std::size_t space = 4 * alignment;
    void* ptr_copy = ptr;
    return (std::align(alignment, size, ptr_copy, space) == ptr);
}

}  // namespace details
}  // namespace vecmem
