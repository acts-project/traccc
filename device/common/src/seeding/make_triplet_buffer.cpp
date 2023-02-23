/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/seeding/device/make_triplet_buffer.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <algorithm>

namespace traccc::device {

triplet_container_types::buffer make_triplet_buffer(
    const std::vector<unsigned int>& triplet_sizes, vecmem::copy& copy,
    vecmem::memory_resource& mr, vecmem::memory_resource* mr_host) {

    const triplet_container_types::buffer::header_vector::size_type
        triplets_size = triplet_sizes.size();

    // Create the result object.
    triplet_container_types::buffer result{{triplets_size, mr},
                                           {triplet_sizes, mr, mr_host}};

    // Initialise the buffer(s).
    copy.setup(result.headers);
    copy.setup(result.items);

    // Make sure that the summary values are set to zero in the buffers.
    copy.memset(result.headers, 0);

    // Return the created buffers.
    return result;
}

}  // namespace traccc::device
