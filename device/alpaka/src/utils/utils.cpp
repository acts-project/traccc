/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "utils.hpp"

namespace traccc::alpaka::details {

Queue get_queue(const traccc::alpaka::queue& q) {

    Queue* queue_ptr = reinterpret_cast<Queue*>(q.alpakaQueue());
    if (queue_ptr == nullptr) {
        throw std::runtime_error("Invalid queue pointer");
    }

    return *queue_ptr;
}

}  // namespace traccc::alpaka::details
