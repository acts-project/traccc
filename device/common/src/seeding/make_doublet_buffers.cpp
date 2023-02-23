/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/seeding/device/make_doublet_buffers.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <algorithm>

namespace traccc::device {

doublet_buffer_pair make_doublet_buffers(
    const std::vector<unsigned int>& mb_sizes,
    const std::vector<unsigned int>& mt_sizes, vecmem::copy& copy,
    vecmem::memory_resource& mr, vecmem::memory_resource* mr_host) {

    const doublet_container_types::buffer::header_vector::size_type mb_size =
        mb_sizes.size();
    const doublet_container_types::buffer::header_vector::size_type mt_size =
        mt_sizes.size();

    // Create the result object.
    doublet_buffer_pair buffers{{{mb_size, mr}, {mb_sizes, mr, mr_host}},
                                {{mt_size, mr}, {mt_sizes, mr, mr_host}}};

    // Initialise the buffer(s).
    copy.setup(buffers.middleBottom.headers);
    copy.setup(buffers.middleBottom.items);
    copy.setup(buffers.middleTop.headers);
    copy.setup(buffers.middleTop.items);

    // Make sure that the summary values are set to zero in the buffers.
    copy.memset(buffers.middleBottom.headers, 0);
    copy.memset(buffers.middleTop.headers, 0);

    // Return the created buffers.
    return buffers;
}

}  // namespace traccc::device
