/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/sycl/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"

namespace traccc::sycl {

silicon_pixel_spacepoint_formation_algorithm::
    silicon_pixel_spacepoint_formation_algorithm(
        const traccc::memory_resource& mr, vecmem::copy& copy,
        queue_wrapper queue, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_mr(mr), m_copy(copy), m_queue(queue) {}

}  // namespace traccc::sycl
