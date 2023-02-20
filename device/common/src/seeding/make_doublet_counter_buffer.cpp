/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/seeding/device/make_doublet_counter_buffer.hpp"

// System include(s).
#include <numeric>

namespace traccc::device {

device::doublet_counter_spM_collection_types::buffer
make_doublet_counter_buffer(const std::vector<unsigned int>& grid_sizes,
                            vecmem::copy& copy, vecmem::memory_resource& mr) {

    // Calculate the capacities for the buffer.
    const device::doublet_counter_spM_collection_types::buffer::size_type size =
        std::accumulate(grid_sizes.begin(), grid_sizes.end(), 0);

    // Create the buffer object.
    device::doublet_counter_spM_collection_types::buffer buffer{size, 0, mr};
    /// TODO: copy.setup(buffer) required here?

    // Make sure that the summary values are set to zero in the buffer.
    copy.memset(buffer, 0);

    // Return the buffer.
    return buffer;
}

}  // namespace traccc::device
