/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/seeding/device/make_doublet_counter_buffer.hpp"

namespace traccc::device {

device::doublet_counter_container_types::buffer make_doublet_counter_buffer(
    const std::vector<unsigned int>& grid_sizes, vecmem::copy& copy,
    vecmem::memory_resource& mr, vecmem::memory_resource* mr_host) {

    // Calculate the capacities for the buffer.
    const device::doublet_counter_container_types::buffer::header_vector::
        size_type buffer_size = grid_sizes.size();

    // Create the buffer object.
    device::doublet_counter_container_types::buffer buffer{
        {buffer_size, mr},
        {std::vector<unsigned int>(buffer_size, 0), grid_sizes, mr, mr_host}};
    copy.setup(buffer.headers);
    copy.setup(buffer.items);

    // Make sure that the summary values are set to zero in the buffer.
    copy.memset(buffer.headers, 0);

    // Return the buffer.
    return buffer;
}

}  // namespace traccc::device
