/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "opaque_queue.hpp"

#include "traccc/alpaka/utils/get_vecmem_resource.hpp"

namespace traccc::alpaka::details {

opaque_queue::opaque_queue(std::size_t device)
    : m_device{device}, m_queue(nullptr) {
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, device);
    m_queue = std::make_unique<Queue>(devAcc);
}

}  // namespace traccc::alpaka::details
