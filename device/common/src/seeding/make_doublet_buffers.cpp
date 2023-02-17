/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
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

doublet_spM_buffer_pair make_doublet_buffers(
    const doublet_counter_spM_collection_types::const_view& doublet_counter,
    vecmem::copy& copy, vecmem::memory_resource& mr,
    vecmem::memory_resource* mr_host) {

    // Get the number of doublets per geometric bin.
    vecmem::vector<device::doublet_counter_spM> doublet_counts(
        (mr_host != nullptr) ? mr_host : &mr);
    copy(doublet_counter, doublet_counts);

    unsigned int size = doublet_counts.size();

    // Construct the size (vectors) for the buffers.
    std::vector<unsigned int> mb_sizes(size);
    std::transform(
        doublet_counts.begin(), doublet_counts.end(), mb_sizes.begin(),
        [](const device::doublet_counter_spM& dc) { return dc.m_nMidBot; });

    std::vector<unsigned int> mt_sizes(size);
    std::transform(
        doublet_counts.begin(), doublet_counts.end(), mt_sizes.begin(),
        [](const device::doublet_counter_spM& dc) { return dc.m_nMidTop; });

    // Create the result object.
    doublet_buffer_pair buffers{{{size, mr}, {mb_sizes, mr, mr_host}},
                                {{size, mr}, {mt_sizes, mr, mr_host}}};

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
