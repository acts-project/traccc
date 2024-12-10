/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/details/memory_resource_base.hpp"

namespace vecmem::details {

bool memory_resource_base::do_is_equal(
    const memory_resource &other) const noexcept {

    // Perform a simple pointer comparison. Assuming that only the very same
    // memory resource objects can be considered equal.
    return (this == &other);
}

}  // namespace vecmem::details
